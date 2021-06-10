import torch
import torch.nn as nn
import torch.nn.functional as F
from function import reward_general
import gumbel_sinkhorn_ops as gs


class SIGMA(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples):
        super(SIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples

    def forward(self, x_s, edge_index_s, x_t, edge_index_t, T, miss_match_value=-1.0):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None

        n_node_s = x_s.shape[0]
        n_node_t = x_t.shape[0]

        theta_previous = None
        for _ in range(T):
            # adjust feature importance
            if theta_previous is not None:
                x_s_importance = theta_previous.sum(-1, keepdims=True)
                x_t_importance = theta_previous.sum(-2, keepdims=True).transpose(-1, -2)
            else:
                x_s_importance = torch.ones([n_node_s, 1]).to(x_s.device)
                x_t_importance = torch.ones([n_node_t, 1]).to(x_s.device)

            # propagate
            h_s = self.f_update(x_s, x_s_importance, edge_index_s)
            h_t = self.f_update(x_t, x_t_importance, edge_index_t)

            # an alternative to the dot product similarity is the cosine similarity.
            # scale 50.0 allows logits to be larger than the noise.
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0

            log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0, n_node_t), value=0.0)

            theta_new = gs.gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            r = reward_general(n_node_s, n_node_t, edge_index_s, edge_index_t, theta_new, miss_match_value=miss_match_value)

            if best_reward < r.item():
                best_reward = r.item()
                loss = loss - r
                loss_count += 1
                theta_previous = theta_new[:, :n_node_s, :n_node_t].mean(0).detach().clone()
                best_logits = log_alpha_.detach().clone()

        return best_logits, loss / float(loss_count)
