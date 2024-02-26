import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


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
        print('c_in', c_in)
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
        print('h.shape', h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class Contrastive_FeatureExtractor_conv(nn.Module):
    def __init__(self, temperature=1, support=None):
        super().__init__()
        self.support = support
        self.temperature = temperature
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
    def forward(self, x):
        print('x.shape', x.shape)
        x = self.conv1(x[:,None,:])
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
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
        print(x[0])
        x = self.gcn(x, self.support)
        print('x.shape_out', x.shape)
        print(x[0])
        return x
    
    def contrast(self, x):
        # project
        x1 = self(x)
        x2 = self(x)
        print('x1.shape', x1.shape)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        # L2 norm
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
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
