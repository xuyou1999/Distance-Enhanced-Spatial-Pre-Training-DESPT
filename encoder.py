import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import torch.fft as fft
from einops import reduce, rearrange
import gc

def print_largest_cuda_tensors(n=10):
    """
    Print details of the largest tensors allocated on CUDA devices.
    
    Parameters:
    - n (int): Number of top tensors to display. Defaults to 10.
    """
    # Function to get the size of a tensor in bytes
    def tensor_size_in_bytes(tensor):
        return tensor.element_size() * tensor.nelement()

    # Collect all tensors that are on CUDA and currently alive
    alive_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda]

    # Sort them by their memory footprint
    sorted_tensors = sorted(alive_tensors, key=lambda x: tensor_size_in_bytes(x), reverse=True)

    # Print the details of the top tensors
    print("Top CUDA tensors by size (bytes):")
    for tensor in sorted_tensors[:n]:  # Display top n tensors
        print(f"Shape: {tensor.shape}, Size (bytes): {tensor_size_in_bytes(tensor)}, Device: {tensor.device}, Type: {tensor.dtype}")

    # If you want to see the total allocated and reserved memory on all CUDA devices
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: Total memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"Device {i}: Total memory reserved: {torch.cuda.memory_reserved(i)} bytes")

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
    def __init__(self, temperature=1, is_gcn=True, is_sampler=False, support_len=1):
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
        self.gcn = gcn(32, 32, 0, support_len, 1)
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
        # print(x[0])
        if self.is_gcn == True:
            x = self.gcn(x, support)
        # print('x.shape_out', x.shape)
        # print(x[0])
        return x
    
    def contrast(self, x1, x2, support1, support2, sensor_idx_start):
        # project
        # print('x.shape', x.shape)
        x1 = self(x1, support1)
        x2 = self(x2, support2)
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
        # input_real = input.real[:, self.start:self.end].to(torch.float32)
        # input_imag = input.imag[:, self.start:self.end].to(torch.float32)
        # weight_real = self.weight.real.to(torch.float32)
        # weight_imag = self.weight.imag.to(torch.float32)

        # # Perform operations on real and imaginary parts separately if needed
        # # Example for einsum, adjust according to your actual computation needs
        # output_real = torch.einsum('bti,tio->bto', input_real, weight_real) - torch.einsum('bti,tio->bto', input_imag, weight_imag)
        # output_imag = torch.einsum('bti,tio->bto', input_real, weight_imag) + torch.einsum('bti,tio->bto', input_imag, weight_real)

        # output = torch.complex(output_real, output_imag)
        weight = self.weight.to(input.device)
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], weight)
        bias = self.bias.to(input.device)
        return output + bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
                 alpha: float = 0.5,
                 temperature: float = 1,
                 is_gcn=True, is_sampler=False, support_len=1
                #  length: int,
                #  hidden_dims=64, depth=10,
                 ):
        super().__init__()
        self.is_gcn = is_gcn
        self.is_sampler = is_sampler
        self.conv1 = torch.nn.Conv1d( 1, 32, 13, stride=1)
        component_dims = output_dims // 2
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.alpha = alpha
        self.temperature = temperature
        self.gcn = gcn(32, 32, 0, support_len, 1)
        self.repr_dropout = nn.Dropout(p=0.1)
        self.kernels = kernels
        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1) for b in range(1)]
        )
        self.temperature = temperature
        self.fc1 = torch.nn.Linear(32*3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.bn3 = torch.nn.BatchNorm1d(32*3)
        self.bn4 = torch.nn.BatchNorm1d(32)

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
        # print('x.shape', x.shape)
        x = self.conv1(x[:,None,:]) # B x Ch x T
        # print('x.shape', x.shape)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
            # print('trend.shape', trend[-1].shape)
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        ).transpose(1, 2)
        # print('trend.shape', trend.shape)

        x = x.transpose(1, 2)  # B x T x Co
        org_device = x.device

        season = []
        for mod in self.sfd:
            if x.device.type == 'mps':
                out = mod(x.to('cpu')).to(org_device)  # b t d
            else:
                out = mod(x)
            season.append(out)
        season = season[0]

        season = self.repr_dropout(season).transpose(1, 2)
        # print('season.shape', season.shape)

        x = torch.cat((trend, season), dim=1)
        
#         print_largest_cuda_tensors()

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
        if self.is_gcn == True:
            x = self.gcn(x, support)
        return x

    def contrast(self, x1, x2, support1, support2, sensor_idx_start):
        # project
        # print('x1.shape', x1.shape)
        x1 = self(x1, support1)
        x2 = self(x2, support2)
        # print('x1.shape', x1.shape)
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