import socket
import os.path as osp

import argparse
import torch
from dataset import SigmaPascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from dgmc_utils import ValidPairDataset
from dgmc import SplineCNN
from model import SIGMA
from function import dgmc_acc
import time

parser = argparse.ArgumentParser()
# dgmc setting
parser.add_argument('--isotropic', action='store_true')
# network setting
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--T', type=int, default=5)
# weight for unsupervised objective
parser.add_argument('--reward_weight', type=float, default=0.5)
parser.add_argument('--cat_hidden', action='store_true')
# gnn activation
parser.add_argument('--gnn_act', type=str, default='leaky_relu')
# training setting
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2norm', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--pretrain_epochs', type=int, default=5)
# gumbel sinkhorn option
parser.add_argument("--tau", default=0.1, type=float, help="temperature parameter")
parser.add_argument("--n_sink_iter", default=10, type=int, help="number of iterations for sinkhorn normalization")
parser.add_argument("--n_samples", default=10, type=int, help="number of samples from gumbel-sinkhorn distribution")
args = parser.parse_args()
print(args)

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join('../data', 'PascalVOC')
for category in PascalVOC.categories:
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
f_update = SplineCNN(dataset.num_node_features, args.dim,
                     dataset.num_edge_features, args.num_layers, cat=args.cat_hidden,
                     dropout=args.dropout, act=args.gnn_act)
model = SIGMA(f_update=f_update, tau=args.tau, n_sink_iter=args.n_sink_iter, n_samples=args.n_samples).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train(reward_weight, use_y=True):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        if use_y:
            y = generate_y(data.y)
        else:
            y = None
        _, loss = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                        data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch,
                        T=args.T, y=y, reward_weight=reward_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(train_loader.dataset)


def test(dataset):
    model.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    correct = num_examples = 0
    for data in loader:
        data = data.to(device)

        S_L, _ = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                       data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch,
                       T=args.T, y=None)

        y = generate_y(data.y)
        correct += dgmc_acc(S_L, y, reduction='sum')
        num_examples += y.size(1)

    return correct / num_examples


for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    if epoch <= args.pretrain_epochs:
        loss = train(reward_weight=1.0, use_y=False)
    else:
        loss = train(reward_weight=args.reward_weight, use_y=True)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    accs = [100 * test(test_dataset) for test_dataset in test_datasets]
    accs += [sum(accs) / len(accs)]

    print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
    print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
