import torch
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from function import reward_general, dgmc_loss
import gumbel_sinkhorn_ops as gs


class SIGMA(torch.nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples):
        super(SIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples

        self.w = torch.nn.Parameter(torch.randn([1, 1, 1, 512]))
        # self.linear = torch.nn.Sequential(torch.nn.Linear(512 * 2, 512 * 2), torch.nn.ReLU(), torch.nn.Linear(512 * 2, 512))

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t,
                T, y=None, reward_weight=1.0):
        loss_r = 0.0
        loss_y = 0.0
        loss_count = 0

        x_s_, x_t_ = x_s, x_t
        r_best = float('-inf')

        _, s_mask = to_dense_batch(x_s, batch_s, fill_value=0)
        _, t_mask = to_dense_batch(x_t, batch_t, fill_value=0)

        n_node_s = s_mask.shape[-1]
        n_node_t = t_mask.shape[-1]

        best_theta = None

        for t in range(T):
            # propagate
            h_s = self.f_update(x_s_, edge_index_s, edge_attr_s)
            h_t = self.f_update(x_t_, edge_index_t, edge_attr_t)

            h_s, _ = to_dense_batch(h_s, batch_s, fill_value=0)
            h_t, _ = to_dense_batch(h_t, batch_t, fill_value=0)

            # h_s = h_s.unsqueeze(-2)
            # h_t = h_t.unsqueeze(-3)
            # h_s = h_s.repeat(1, 1, h_t.shape[-2], 1)
            # h_t = h_t.repeat(1, h_s.shape[-3], 1, 1)
            # h = torch.cat([h_s, h_t], dim=-1)
            # w = self.linear(h)
            # log_alpha = (h_s * h_t * torch.sigmoid(w)).sum(-1)
            # log_alpha = (h_s * h_t * torch.sigmoid(self.w)).sum(-1)
            log_alpha = h_s @ h_t.transpose(-1, -2)

            log_alpha_mask = s_mask.unsqueeze(-1) & t_mask.unsqueeze(-2)
            log_alpha = log_alpha.masked_fill(~log_alpha_mask, min(log_alpha.min().item()-5.*abs(log_alpha.min().item()), -1000.))

            log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0, n_node_t), value=0.0)

            theta_new = gs.gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            theta_new = theta_new[:, :, :n_node_s, :n_node_t]

            r = reward_general(edge_index_s, edge_index_t, batch_s, batch_t, theta_new, s_mask, t_mask)

            if r.item() > r_best:
                r_best = r.item()

                # feature importance
                theta_new_detach_copy = theta_new.detach().clone()
                x_s_ = x_s * theta_new_detach_copy.mean(1).sum(-1, keepdims=True)[s_mask]
                x_t_ = x_t * theta_new_detach_copy.mean(1).sum(-2, keepdims=True).transpose(-1, -2)[t_mask]

                # add negative reward to loss
                loss_r = loss_r - reward_weight * r
                loss_count += 1

                # for prediction loss
                theta_new = theta_new.transpose(0, 1)
                theta_new = theta_new[:, s_mask]
                if y is not None:
                    loss_y = loss_y + dgmc_loss(theta_new, y)

                # follow dgmc and return the matching matrix for evaluation
                best_theta = gs.gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, 1, noise=False).detach()
                best_theta = best_theta[:, :, :n_node_s, :n_node_t].transpose(0, 1)[:, s_mask].mean(0)

        # loss to train
        loss = loss_r / float(loss_count) + loss_y / float(loss_count)

        return best_theta, loss


