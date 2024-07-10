import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from typing import List
import torch.fft as fft
from einops import reduce, rearrange
import gc

def nt_xent_loss(out_1, out_2, temperature):
    """
    Loss used in SimCLR.
    InfoNCE loss
    Ref: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py
    """
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
    # total allocated and reserved memory on all CUDA devices
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: Total memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"Device {i}: Total memory reserved: {torch.cuda.memory_reserved(i)} bytes")


# ------------------- SCPT -------------------
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        '''
        Apply matrix A to the input x by a matrix multiplication
        No parameter to learn at this stage
        '''
        x = torch.einsum('vl,vw->wl',(x,A)) # (V,l) x (V,V) -> (V,l)
        return x.contiguous()

class gcn(nn.Module):
    # Graph Convolutional Network
    def __init__(self,c_in,c_out,dropout,support_len=3,order=1):
        '''
        support length: number of adjacency matrix to consider
        order: number of neighbor steps to consider
        c_in: number of input channel for each sensor, to the mlp
        c_out: number of output channel for each sensor, out from the mlp
        '''
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        '''
        For each adjacency matrix in the support, apply the graph convolution to the input x
        For each additional order, apply the graph convolution to the previous output
        All outputs (include original input x) is added input a out list
        Then concatenate all outputs in the list and apply a mlp
        '''
        out = [x] # [[V,l]]
        for a in support: # a is the adjacency matrix [V,V]
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1) # [V,(1+order*support_len)*l]
        h = self.mlp(h)
        # dropout applied only during training time
        h = F.dropout(h, self.dropout, training=self.training) 
        return h
    
class Contrastive_FeatureExtractor_conv(nn.Module):
    def __init__(self, temperature, is_gcn, is_sampler, support_len, gcn_order, gcn_dropout):
        super().__init__()
        self.temperature = temperature
        self.is_gcn = is_gcn
        self.conv1 = torch.nn.Conv1d( 1, 32, 13, stride=1) 
        self.conv2 = torch.nn.Conv1d(32, 32, 12, stride=12)
        self.conv3 = torch.nn.Conv1d(32, 32, 24, stride=24)
        self.fc1 = torch.nn.Linear(32*3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32*3)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.gcn = gcn(32, 32, gcn_dropout, support_len, gcn_order)
        self.is_sampler = is_sampler

    def forward(self, x, support, is_example):
        if is_example:
            print('\nEncoder started')
        x = self.conv1(x[:,None,:])
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        if is_example == False:
            x = F.relu(x)
            x = self.bn2(x)
            x = self.conv3(x)
        '''
        If the sampler is enabled, the input x is sampled by half of latent features
        '''
        if self.is_sampler == False:
            x_ = x
        else:
        # sample half of features of each sensor's representation
            n_half = int(x.shape[-1]/2)
            x_ = torch.empty(x.shape[0], x.shape[1], n_half).to(x.device)
            for i in range(x.shape[0]):
                idx = np.arange(x.shape[2])
                np.random.shuffle(idx)
                idx = idx < n_half
                x_[i, :, :] = x[i, :, idx]
        if is_example:
            print('\nthe data shape after convolutional layers and potential sampling')
            print('x_.shape', x_.shape)
        # aggregate
        x_u = x_.mean(axis=2)
        x_z = x_.std(axis=2)
        x_x, _ = torch.max(x_, axis=2)
        x = torch.cat((x_u, x_z, x_x), axis=1)

        x = self.bn3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)

        # GCN
        if self.is_gcn == True:
            if is_example:
                print('GCN is enabled')
            x = self.gcn(x, support)
        if is_example:
            print('\nEncoder finished')
        return x
    
    def contrast(self, x1, x2, support1, support2, sensor_idx_start, is_example):
        '''
        x1 and x2 are two input samples
        support1 and support2 are adjacency matrix for each sample
        An additional projection head is applied to conduct loss calculation
        '''
        x1 = self(x1, support1, is_example)
        x2 = self(x2, support2, is_example)
        # Projection
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        # Only evaluate the sensors after sensor_idx_start index
        x1 = x1[sensor_idx_start:]
        x2 = x2[sensor_idx_start:]
        if is_example:
            print('\nIn performing encoder contrast learning, the data shape before loss calculation:')
            print('x1.shape', x1.shape)
            print('x2.shape', x2.shape)
        # calculate loss
        return nt_xent_loss(x1,x2,self.temperature)
    

# ------------------- CoST -------------------
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        def custom_forward(x):
            residual = x if self.projector is None else self.projector(x)
            x = F.gelu(x)
            x = self.conv1(x)
            x = F.gelu(x)
            x = self.conv2(x)
            return x + residual
        
        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, extract_layers=None):
        super().__init__()

        if extract_layers is not None:
            assert len(channels) - 1 in extract_layers

        self.extract_layers = extract_layers
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        if self.extract_layers is not None:
            outputs = []
            for idx, mod in enumerate(self.net):
                x = mod(x)
                if idx in self.extract_layers:
                    outputs.append(x)
            return outputs
        return self.net(x)

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
                 kernels, length, hidden_dims, depth,
                 alpha, temperature,
                 is_gcn, is_sampler, support_len, 
                 gcn_order, gcn_dropout
                 ):
        super().__init__()
        self.is_gcn = is_gcn
        self.is_sampler = is_sampler
        component_dims = output_dims // 2
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.alpha = alpha
        self.temperature = temperature
        self.gcn_trend = gcn(16, 16, gcn_dropout, support_len, gcn_order)
        self.gcn_season = gcn(16, 16, gcn_dropout, support_len, gcn_order)
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.kernels = kernels
        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )
        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )
        self.temperature = temperature
        self.fc1_trend = torch.nn.Linear(16*3, 16)
        self.fc1_season = torch.nn.Linear(16*3, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.bn3_trend = torch.nn.BatchNorm1d(16*3)
        self.bn3_season = torch.nn.BatchNorm1d(16*3)
        self.bn4_trend = torch.nn.BatchNorm1d(16)
        self.bn4_season = torch.nn.BatchNorm1d(16)
    
    def forward(self, x, support, is_example):
        x = self.input_fc(x[:,:,None]) # B x T x Co
        x = x.transpose(1, 2)
        x = self.feature_extractor(x) # B x Co x T
        
        # Trend component
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

        # Sampling and aggregation for trend component
        if self.is_sampler == False:
            trend_ = trend
        else:
            n_half = int(trend.shape[-1]/2)
            trend_ = torch.empty(trend.shape[0], trend.shape[1], n_half).to(trend.device)
            for i in range(trend.shape[0]):
                idx = np.arange(trend.shape[2])
                np.random.shuffle(idx)
                idx = idx < n_half
                trend_[i, :, :] = trend[i, :, idx]
        trend_u = trend_.mean(axis=2)
        trend_z = trend_.std(axis=2)
        trend_x, _ = torch.max(trend_, axis=2)
        trend = torch.cat((trend_u, trend_z, trend_x), axis=1)
        trend = self.bn3_trend(trend)
        trend = self.fc1_trend(trend)
        trend = F.relu(trend)
        trend = self.bn4_trend(trend)
        if self.is_gcn == True:
            trend = self.gcn_trend(trend, support)
        
        # Seasonal component
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

        # Sampling and aggregation for seasonal component
        if self.is_sampler == False:
            season_ = season
        else:
            n_half = int(season.shape[-1]/2)
            season_ = torch.empty(season.shape[0], season.shape[1], n_half).to(season.device)
            for i in range(season.shape[0]):
                idx = np.arange(season.shape[2])
                np.random.shuffle(idx)
                idx = idx < n_half
                season_[i, :, :] = season[i, :, idx]
        season_u = season_.mean(axis=2)
        season_z = season_.std(axis=2)
        season_x, _ = torch.max(season_, axis=2)
        season = torch.cat((season_u, season_z, season_x), axis=1)
        season = self.bn3_season(season)
        season = self.fc1_season(season)
        season = F.relu(season)
        season = self.bn4_season(season)
        if self.is_gcn == True:
            season = self.gcn_season(season, support)

        # Concatenate trend and seasonal components
        x = torch.cat((trend, season), dim=1)

        return x

    def contrast(self, x1, x2, support1, support2, sensor_idx_start, is_example):
        x1 = self(x1, support1, is_example)
        x1_trend = x1[:,:16]
        x1_season = x1[:,16:]
        x2 = self(x2, support2, is_example)
        x2_trend = x2[:,:16]
        x2_season = x2[:,16:]
        # projection
        print('x1_trend.shape', x1_trend.shape)
        print('x1_season.shape', x1_season.shape)
        x1_trend = self.fc2(x1_trend)
        x1_season = self.fc2(x1_season)
        x2_trend = self.fc2(x2_trend)
        x2_season = self.fc2(x2_season)
        # L2 norm
        x1_trend = F.normalize(x1_trend)
        x1_season = F.normalize(x1_season)
        x2_trend = F.normalize(x2_trend)
        x2_season = F.normalize(x2_season)
        x1_trend = x1_trend[sensor_idx_start:]
        x1_season = x1_season[sensor_idx_start:]
        x2_trend = x2_trend[sensor_idx_start:]
        x2_season = x2_season[sensor_idx_start:]
        # calculate loss
        trend_loss = nt_xent_loss(x1_trend, x2_trend, self.temperature)
        season_loss = nt_xent_loss(x1_season, x2_season, self.temperature)
        final_loss = trend_loss + self.alpha * season_loss
        return final_loss


class CoSTEncoder_legacy(nn.Module):
    def __init__(self, input_dims, output_dims, kernels,alpha, temperature,
                 is_gcn, is_sampler, support_len, gcn_order, gcn_dropout
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
        self.gcn = gcn(32, 32, gcn_dropout, support_len, gcn_order)
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
    
    def forward(self, x, support, is_example):  # x: B x T x input_dims
        x = self.conv1(x[:,None,:]) # B x Ch x T

        # Trend component
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

        # Seasonal component
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

        # Concatenate trend and seasonal components
        x = torch.cat((trend, season), dim=1)

        if self.is_sampler == False:
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
        x = self.bn3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn4(x)
        if self.is_gcn == True:
            x = self.gcn(x, support)
        return x

    def contrast(self, x1, x2, support1, support2, sensor_idx_start, is_example):
        x1 = self(x1, support1, is_example)
        x2 = self(x2, support2, is_example)
        # projection
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # L2 norm
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        x1 = x1[sensor_idx_start:]
        x2 = x2[sensor_idx_start:]
        # calculate loss
        return nt_xent_loss(x1,x2,self.temperature)