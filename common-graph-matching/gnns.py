import torch
from torch.nn import Sequential, Linear, Tanh
from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv1 = GINConv(self.nn1, eps=eps, train_eps=train_eps)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv2 = GINConv(self.nn2, eps=eps, train_eps=train_eps)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINConv(self.nn3, eps=eps, train_eps=train_eps)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINConv(self.nn4, eps=eps, train_eps=train_eps)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINConv(self.nn5, eps=eps, train_eps=train_eps)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, X_importance, edge_index):
        x = self.bn_in(X * X_importance)
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index))
        xs.append(self.conv2(xs[-1], edge_index))
        xs.append(self.conv3(xs[-1], edge_index))
        xs.append(self.conv4(xs[-1], edge_index))
        xs.append(self.conv5(xs[-1], edge_index))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x

