'''
GraphWaveNet.py
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np
import pandas as pd

class gwnet_nconv(nn.Module):
    def __init__(self):
        super(gwnet_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class gwnet_linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(gwnet_linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gwnet_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gwnet_gcn,self).__init__()
        self.nconv = gwnet_nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = gwnet_linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # print('shape of x is:', x.shape)
        # print('shape of support is:', len(support))
        out = [x]
        for a in support:
            # print('shape of a is:', a.shape)
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        # print('shape of out is:', len(out))
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        # print('shape of h is:', h.shape)
        return h

class gwnet(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 dropout=0.0,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 sga = True,
                 adp_adj = False):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adp_adj = adp_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # SGA: spatially gated addition
        if sga:
            self.sga_gate1 = nn.Conv2d(in_channels=2*residual_channels, out_channels=4*residual_channels, kernel_size=(1,1))
            self.sga_gate2 = nn.Conv2d(in_channels=4*residual_channels, out_channels=1, kernel_size=(1,1))
            self.addition = self.spatially_gated_addition
        else:
            self.addition = self.naive_addition

        # linear projections from embedding to adaptive adjacency
        if adp_adj:
            self.embed_proj11 = nn.Linear(residual_channels,4*residual_channels, bias=False)
            self.embed_proj12 = nn.Linear(residual_channels*4,residual_channels, bias=False)
            self.embed_proj21 = nn.Linear(residual_channels,4*residual_channels, bias=False)
            self.embed_proj22 = nn.Linear(residual_channels*4,residual_channels, bias=False)
    
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                # *****************************
                # VERY IMPORTANT: adjust the adjacency matrix length accordingly!!!!!!
                self.gconv.append(gwnet_gcn(dilation_channels,residual_channels,dropout,support_len=1 + int(adp_adj)))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def spatially_gated_addition(self, x, e):
        # x [BDNL] is latent representation
        # e [DN] is embedding
        # only one set, because this is just a quality gate for the embedding
        # they all should have the same quality.
        e = e.unsqueeze(0).unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1]) # BDNL
        g = self.sga_gate1(torch.cat((x,e), dim=1))
        g = F.relu(g)
        g = self.sga_gate2(g)
        g = torch.sigmoid(g)
        x = x + g * e
        return x

    def naive_addition(self, x, e):
        return x + e.unsqueeze(0).unsqueeze(-1).expand(x.shape[0],-1,-1,x.shape[-1]) # BDNL

    def forward(self, input, adj, embed):
        # print('shape of input is:', input.shape)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # adaptive adjacency based on SCPT embeddings
        if self.adp_adj:
            # without SCPT, embed=0 anyway, since the projs have no bias, then it is all zeroes anyway.
            # nv = nodevec
            nv1 = self.embed_proj12(F.relu(self.embed_proj11(embed.T))) # embed = [D,N]
            nv2 = self.embed_proj22(F.relu(self.embed_proj21(embed.T))) # embed = [D,N]
            adp = F.softmax(F.relu(torch.mm(nv1, nv2.T)), dim=1) # adp = [N,N]
            adj = adj + [adp]

        # WaveNet layers
        # print('shape of x is:', x.shape)
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            x = self.addition(x, embed)
            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # GCN
            x = self.addition(x, embed)
            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x