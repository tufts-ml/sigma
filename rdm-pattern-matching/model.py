import torch
import torch.nn as nn
import torch.nn.functional as F
from function import reward_rdm
import gumbel_sinkhorn_ops as gs


class SIGMA(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples, rw_base, rw_rdm):
        super(SIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples
        self.rw_base = rw_base
        self.rw_rdm = rw_rdm

    def forward(self, x_s, x_s_type, edge_index_s, edge_attr_s, x_t, x_t_type, edge_index_t, edge_attr_t, rdm, T):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None
        theta_previous = None

        n_node_s = x_s.shape[0]
        n_node_t = x_t.shape[0]
        for t in range(T):
            # adjust feature importance
            if theta_previous is not None:
                x_s_importance = theta_previous.sum(-1, keepdims=True)
                x_t_importance = theta_previous.sum(-2, keepdims=True).transpose(-1, -2)
            else:
                x_s_importance = torch.ones([n_node_s, 1]).to(x_s.device)
                x_t_importance = torch.ones([n_node_t, 1]).to(x_t.device)

            # propagate
            h_s = self.f_update(x_s, x_s_importance, edge_index_s, edge_attr_s, 1)
            h_t = self.f_update(x_t, x_t_importance, edge_index_t, edge_attr_t, 2)

            logits = h_s @ h_t.transpose(-1, -2)
            logits = F.pad(logits, (0, n_node_s, 0, n_node_t))

            theta_new = gs.gumbel_sinkhorn(logits, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            r = reward_rdm(x_s_type, x_t_type, edge_index_s, edge_index_t, rdm, theta_new, self.rw_base, self.rw_rdm)

            if best_reward < r.item():
                best_reward = r.item()
                loss = loss - r
                loss_count += 1
                best_logits = logits[:n_node_s, :n_node_t].detach().clone()
                theta_previous = theta_new[:, :n_node_s, :n_node_t].mean(0).detach().clone()

        return best_logits, loss / float(loss_count)


class AtomEncoderWithPosition(nn.Module):
    def __init__(self, nt_m, np_m, emb_dim):
        """
        :param nt: node type.
        :param np: node position.
        """
        super(AtomEncoderWithPosition, self).__init__()

        self.nt_m = nt_m
        self.np_m = np_m

        self.nt_emb = nn.Embedding(nt_m, emb_dim)
        nn.init.xavier_uniform_(self.nt_emb.weight.data)

        self.np_emb = nn.Sequential(
            nn.Linear(np_m, emb_dim),
        )

    def forward(self, X):
        # split input to node type and node position
        X_nt = X[:, : self.nt_m]
        X_np = X[:, self.nt_m: self.nt_m + self.np_m]

        # node type embedding
        # convert one-hot to indices
        X_nt_int = X_nt @ torch.arange(self.nt_m, dtype=torch.float).view(-1, 1).to(X_nt.device)
        X_nt_int = X_nt_int.view(-1).long()
        h_nt = self.nt_emb(X_nt_int)

        # node position embedding
        h_np = self.np_emb(X_np)

        h = torch.cat([h_nt, h_np], dim=-1)

        return h
