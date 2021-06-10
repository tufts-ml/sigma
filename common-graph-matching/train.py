import argparse
import pickle

import torch
import numpy as np
from model import SIGMA
from gnns import GIN
from function import predict


def train(p_s, cost_s, p_t, cost_t):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    for epoch in (range(args.epochs)):
        # forward model
        model.train()
        _, loss = model(p_s, cost_s, p_t, cost_t, args.T, miss_match_value=args.miss_match_value)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        # evaluate
        with torch.no_grad():
            model.eval()
            logits_t, _ = model(p_s, cost_s, p_t, cost_t, args.T, miss_match_value=args.miss_match_value)
            evaluate(logits_t, epoch)


def evaluate(log_alpha, epoch=0):
    matched_row, matched_col = predict(log_alpha, n=1004, m=1004)

    pair_names = []
    for i in range(matched_row.shape[0]):
        pair_names.append([idx2node_s[matched_row[i]], idx2node_t[matched_col[i]]])

    node_correctness = 0
    for pair in pair_names:
        if pair[0] == pair[1]:
            node_correctness += 1
    node_correctness /= num_nodes

    print('Epoch: %d, NC: %.1f' % (epoch+1, node_correctness * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset option
    parser.add_argument('--data_path', type=str, default='../data/ppi.pkl')
    # network option
    parser.add_argument('--node_feature_dim', type=int, default=1)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--miss_match_value', type=float, default=0.01)
    # training option
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2norm', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperature parameter")
    parser.add_argument("--n_sink_iter", default=10, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=10, type=int, help="number of samples from gumbel-sinkhorn distribution")
    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # we recommend to use gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def pp(cost_in):
        cost = np.array(cost_in.todense())
        p = cost.sum(-1, keepdims=True)
        p = p / p.sum()
        p = torch.FloatTensor(p).to(device)

        cost = np.where(cost != 0)
        cost = torch.LongTensor(cost).to(device)

        return p, cost

    for gi in range(0, 1):
        # construct model
        f_update = GIN(in_channels=args.node_feature_dim, out_channels=args.dim, dim=args.dim)
        model = SIGMA(f_update, tau=args.tau, n_sink_iter=args.n_sink_iter, n_samples=args.n_samples).to(device)

        cost_s = data['costs'][0]
        cost_t = data['costs'][gi + 1]
        p_s = data['probs'][0]
        p_t = data['probs'][gi + 1]
        idx2node_s = data['idx2nodes'][0]
        idx2node_t = data['idx2nodes'][gi + 1]
        num_nodes = min([len(idx2node_s), len(idx2node_t)])

        p_s, cost_s = pp(cost_s)
        p_t, cost_t = pp(cost_t)

        train(p_s, cost_s, p_t, cost_t)
