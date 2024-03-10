import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch.fft as fft
from einops import reduce, rearrange, repeat


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('vl,vw->wl',(x,A)) # (N,C,V,l) x (V,V) -> (N,C,V,l)
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order # hwo many neighbor steps to consider

    def forward(self,x,support):
        out = [x]
        for a in support: # a is the adjacency matrix [V,V]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        # print('h.shape', h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class Contrastive_FeatureExtractor_conv(nn.Module):
    def __init__(self, temperature=1, is_gcn=True, is_sampler=False):
        super().__init__()
        self.temperature = temperature
        self.is_gcn = is_gcn
        self.conv1 = torch.nn.Conv1d( 1, 32, 13, stride=1) # 1 hour --> per timestep
        self.conv2 = torch.nn.Conv1d(32, 32, 12, stride=12) # 2 hour --> per hour
        self.conv3 = torch.nn.Conv1d(32, 32, 24, stride=24) # 1 day --> per day
        self.fc1 = torch.nn.Linear(32*3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32*3)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.gcn = gcn(32, 32, 0, 1, 1)
        self.is_sampler = is_sampler
    def forward(self, x, support):
        # print('x.shape', x.shape)
        x = self.conv1(x[:,None,:])
        # print('x.shape', x.shape)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        if self.is_gcn == True and self.is_sampler == False:
            x_ = x
        else:
        # sample half of samples
            n_half = int(x.shape[-1]/2)
            x_ = torch.empty(x.shape[0], x.shape[1], n_half).to(x.device)
            for i in range(x.shape[0]):
                idx = np.arange(x.shape[2])
                np.random.shuffle(idx)
                idx = idx < n_half
                x_[i, :, :] = x[i, :, idx]
        # aggregate
        x_u = x_.mean(axis=2)
        x_z = x_.std(axis=2)
        x_x, _ = torch.max(x_, axis=2)
        x = torch.cat((x_u, x_z, x_x), axis=1)
        # project
        x = self.bn3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)
        # print(x.shape)
        if self.is_gcn == True:
            x = self.gcn(x, support)
        # print('x.shape_out', x.shape)
        # print(x[0])
        return x
    
    def contrast(self, x, support1, support2, sensor_idx_start):
        # project
        # print('x.shape', x.shape)
        x1 = self(x, support1)
        x2 = self(x, support2)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # L2 norm
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x1 = x1[sensor_idx_start:]
        x2 = x2[sensor_idx_start:]
        # print('x1.shape', x1.shape)
        # calculate loss
        return nt_xent_loss(x1,x2,self.temperature)
    

def nt_xent_loss(out_1, out_2, temperature):
    """Loss used in SimCLR."""
    # https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss

# ------------------- CoST -------------------
class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        # Parameters to learn
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))

        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        # print(output_fft.shape, 'output_fft')
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class CoSTEncoder_first(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
                 alpha: float = 0.5,
                 temperature: float = 1,
                #  length: int,
                #  hidden_dims=64, depth=10,
                #  mask_mode='binomial'
                 ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d( 1, 32, 13, stride=1)

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.alpha = alpha
        self.temperature = temperature
        self.gcn = gcn(32, 32, 0, 1, 1)
        # self.hidden_dims = hidden_dims
        # self.mask_mode = mask_mode
        # self.input_fc = nn.Linear(input_dims, hidden_dims)

        # self.feature_extractor = DilatedConvEncoder(
        #     hidden_dims,
        #     [hidden_dims] * depth + [output_dims],
        #     kernel_size=3
        # )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1) for b in range(1)]
        )

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase
    
    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss
    
    def forward(self, x, support):  # x: B x T x input_dims
        gcn_model = gcn(32, 32, 0, 1, 1)
        x = self.gcn(x, support)
        x = self.conv1(x[:,None,:]) # B x Ch x T
        
        # print(x.shape)

        # # conv encoder
        # x = self.feature_extractor(x)  # B x Co x T

        # if tcn_output:
        #     return x.transpose(1, 2)
        # print(x.shape)
        # print(support[0].shape)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        ).transpose(1, 2)
        # print(trend.shape)

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]
        # print(self.repr_dropout(season).shape)

        season = self.repr_dropout(season).transpose(1, 2)

        return trend, season
    
    def contrast(self, x, support1, support2, sensor_idx_start):
        # project
        # print('x.shape', x.shape)
        x1_t, x1_s = self(x, support1)
        x1_t = x1_t.transpose(1, 2)
        x1_s = x1_s.transpose(1, 2)
        x2_t, x2_s = self(x, support2)
        x2_t = x2_t.transpose(1, 2)
        x2_s = x2_s.transpose(1, 2)
        # print(x1_t.shape, x2_t.shape)
        trend_loss = self.instance_contrastive_loss(x1_t, x2_t)
        # print(x1_s.shape, x2_s.shape)
        x1_s = x1_s[sensor_idx_start:]
        x2_s = x2_s[sensor_idx_start:]
        # print(x1_s.shape, x2_s.shape)

        x1_freq = fft.rfft(x1_s, dim=1)
        x2_freq = fft.rfft(x2_s, dim=1)
        # print(x1_freq.shape, x2_freq.shape)
        x1_amp, x1_phase = self.convert_coeff(x1_freq)
        x2_amp, x2_phase = self.convert_coeff(x2_freq)

        seasonal_loss = self.instance_contrastive_loss(x1_amp, x2_amp) + \
                        self.instance_contrastive_loss(x1_phase,x2_phase)

        loss = (self.alpha * (seasonal_loss/2)) + trend_loss
        print(loss)


        return loss
    
class CoSTEncoder_second(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.fc1 = torch.nn.Linear(32*3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.bn3 = torch.nn.BatchNorm1d(32*3)
        self.bn4 = torch.nn.BatchNorm1d(32)
    def forward(self, x):
        x_ = x
        # aggregate
        x_u = x_.mean(axis=2)
        x_z = x_.std(axis=2)
        x_x, _ = torch.max(x_, axis=2)
        x = torch.cat((x_u, x_z, x_x), axis=1)
        # project
        x = self.bn3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)
        # print('x.shape_out', x.shape)
        # print(x[0])
        return x
    
    def contrast(self, x1, x2, sensor_idx_start):
        # project
        # print('x.shape', x.shape)
        x1 = self(x1)
        x2 = self(x2)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # L2 norm
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x1 = x1[sensor_idx_start:]
        x2 = x2[sensor_idx_start:]
        # print('x1.shape', x1.shape)
        # calculate loss
        return nt_xent_loss(x1,x2,self.temperature)