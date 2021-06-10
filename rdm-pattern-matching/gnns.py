import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import FastRGCNConv


class RGCN(torch.nn.Module):
    def __init__(self, node_encoder, in_channels, out_channels, dims, num_layers, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()
        self.node_encoder = node_encoder

        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers-1):
            self.batch_norms.append(BatchNorm(dims))
        self.batch_norms.append(BatchNorm(out_channels))

        base_gnn = FastRGCNConv
        self.convs = nn.ModuleList()
        self.convs.append(base_gnn(in_channels, dims, num_relations, num_bases=num_bases))
        for _ in range(1, num_layers-1):
            self.convs.append(base_gnn(dims, dims, num_relations, num_bases=num_bases))
        self.convs.append(base_gnn(dims, out_channels, num_relations, num_bases=num_bases))

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, X, X_importance, edge_index, edge_type, index):
        x = self.node_encoder(X)
        x = x * X_importance

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = F.relu(x)
            x = self.batch_norms[i](x)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
