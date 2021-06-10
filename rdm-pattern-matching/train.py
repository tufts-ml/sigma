import random
import argparse

import torch
import numpy as np
from model import SIGMA, AtomEncoderWithPosition
from gnns import RGCN
from function import predict, matched_rdm

import pickle

EPS = 1e-6


def distribute_input(key):
    c1, c2 = key.split('_')
    value = data[key]

    x_s, edge_index_s, edge_attr_s = value[c1]['node_feature'], value[c1]['bond'], value[c1]['btype']
    x_t, edge_index_t, edge_attr_t = value[c2]['node_feature'], value[c2]['bond'], value[c2]['btype']
    rdm = value['rdm']

    return x_s.to(device), edge_index_s.to(device), edge_attr_s.to(device), x_t.to(device), edge_index_t.to(
        device), edge_attr_t.to(device), rdm


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    train_ids = list(data.keys())
    for epoch in range(args.epochs):
        model.train()

        random.shuffle(train_ids)

        for itr, key in enumerate(train_ids):
            # construct input
            x_s, edge_index_s, edge_attr_s, x_t, edge_index_t, edge_attr_t, rdm = distribute_input(key)

            # forward model
            _, loss = model(x_s, x_s[:, :args.node_type_dim], edge_index_s, edge_attr_s,
                            x_t, x_t[:, :args.node_type_dim], edge_index_t, edge_attr_t,
                            rdm, args.T)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_s, edge_index_s, edge_attr_s, x_t, edge_index_t, edge_attr_t, rdm

        evaluate(epoch)


def evaluate(epoch=0):
    with torch.no_grad():
        model.eval()

        test_ids = data.keys()
        rdm_matched_ratio = []
        rdm_matched_individual_ratio = []
        for itr, key in enumerate(test_ids):
            # construct input
            x_s, edge_index_s, edge_attr_s, x_t, edge_index_t, edge_attr_t, rdm = distribute_input(key)

            # forward model
            logits, _ = model(x_s, x_s[:, :args.node_type_dim], edge_index_s, edge_attr_s,
                              x_t, x_t[:, :args.node_type_dim], edge_index_t, edge_attr_t,
                              None, args.T)

            matched_row, matched_col = predict(logits)

            n, m = logits.shape

            rdm_matched_ratio_ = matched_rdm(n, m, matched_row, matched_col, rdm, soft=False)
            rdm_matched_individual_ratio_ = matched_rdm(n, m, matched_row, matched_col, rdm, soft=True)

            rdm_matched_ratio.append(rdm_matched_ratio_)
            rdm_matched_individual_ratio.append(rdm_matched_individual_ratio_)

    print('Epoch: %d, hard match: %.1f, soft match %.1f' %
          (epoch + 1, 100 * np.mean(rdm_matched_ratio), 100 * np.mean(rdm_matched_individual_ratio)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset option
    parser.add_argument('--data_path', type=str, default='../data/rdm.pkl')
    parser.add_argument('--node_type_dim', type=int, default=69)
    parser.add_argument('--edge_type_dim', type=int, default=3)
    parser.add_argument('--node_position_dim', type=int, default=50)
    # network option
    parser.add_argument('--node_feature_dim', type=int, default=256)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--rw_base', type=float, default=1.0)
    parser.add_argument('--rw_rdm', type=float, default=1.0)
    # training option
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2norm', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperature parameter")
    parser.add_argument("--n_sink_iter", default=10, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=10, type=int, help="number of samples from gumbel-sinkhorn distribution")
    args = parser.parse_args()

    print(args)

    with open(args.data_path, 'rb') as fi:
        data = pickle.load(fi)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # construct model
    node_encoder = AtomEncoderWithPosition(nt_m=args.node_type_dim, np_m=args.node_position_dim,
                                           emb_dim=args.node_feature_dim).to(device)
    f_update = RGCN(node_encoder,
                    in_channels=args.node_feature_dim * 2, out_channels=args.node_feature_dim,
                    dims=args.node_feature_dim,
                    num_layers=args.num_layers, num_relations=args.edge_type_dim, num_bases=args.num_bases,
                    dropout=args.dropout).to(device)
    model = SIGMA(f_update=f_update, tau=args.tau, n_sink_iter=args.n_sink_iter, n_samples=args.n_samples,
                  rw_base=args.rw_base, rw_rdm=args.rw_rdm).to(device)

    train()
