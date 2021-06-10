import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import SplineConv

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

EPS = 1e-8


class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True,
                 lin=True, dropout=0.0, act=None):
        super(SplineCNN, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SplineConv(in_channels, out_channels, dim=2, kernel_size=5, aggr="max")
            self.convs.append(conv)
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        xs = [x]

        for i, conv in enumerate(self.convs):
            xs += [self.act(conv(xs[-1], edge_index, edge_attr))]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x

        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.dim, self.num_layers, self.cat,
                                      self.lin, self.dropout)
