import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def reward_rdm(n1, n2, e1, e2, rdm, theta, rw_base, rw_rdm, **kwargs):
    n_node_1 = n1.shape[0]
    n_node_2 = n2.shape[0]

    if rw_base > 0.0:
        # node
        nm = (n1 @ n2.transpose(-1, -2)).unsqueeze(0)
        r_node = (nm * theta[:, :n_node_1, :n_node_2]).sum(dim=(-1, -2)).mean()

        # edge
        sn_a1 = e1.shape[1]
        sn_a2 = e2.shape[1]
        sparse_ids = [e1[0].view([-1, 1]).repeat(1, sn_a2).view(-1), e1[1].view([-1, 1]).repeat(1, sn_a2).view(-1),
                      e2[0].repeat(sn_a1), e2[1].repeat(sn_a1)]
        sparse_s1 = theta[:, sparse_ids[0], sparse_ids[2]]
        sparse_s2 = theta[:, sparse_ids[1], sparse_ids[3]]
        r_edge = (sparse_s1 * sparse_s2).sum(dim=-1).mean()
    else:
        r_node, r_edge = 0.0, 0.0

    # rdm
    r_rdm = 0.0
    if rdm is not None and rw_rdm > 0.0:
        theta_nodes = theta[:, :n_node_1 + 1, :n_node_2 + 1].clone()
        theta_nodes[:, :, -1] = theta[:, :n_node_1 + 1, n_node_2:].sum(dim=-1)
        theta_nodes[:, -1, :] = theta[:, n_node_1:, :n_node_2 + 1].sum(dim=-2)
        for rdm_ in rdm:
            r_rdm_ = torch.log(theta_nodes[:, rdm_[:, 0], rdm_[:, 1]]+1e-20)
            r_rdm = r_rdm + r_rdm_.sum(dim=-1).mean()

    # total reward
    r = rw_base * (r_node + r_edge) + rw_rdm * r_rdm

    return r


def predict(log_alpha):
    n, m = log_alpha.shape
    log_alpha = F.pad(log_alpha, (0, n, 0, m))
    log_alpha = log_alpha.cpu().detach().numpy()

    row, col = linear_sum_assignment(-log_alpha)
    matched_mask = (row < n) & (col < m)
    row = row[matched_mask]
    col = col[matched_mask]

    return row, col


def matched_rdm(n, m, row, col, rdms, soft):
    count = 0

    row_ = -1 * np.ones(n+1)
    row_[row] = col
    col_ = -1 * np.ones(m+1)
    col_[col] = row

    for rdm in rdms:
        rdm = rdm[:, :2].cpu().detach().numpy()

        rdm_0 = rdm[:, 0]
        rdm_1 = rdm[:, 1]
        # apply ``or'' to match difference atoms
        matched = np.logical_or(row_[rdm_0] == rdm_1, col_[rdm_1] == rdm_0)
        if not soft:
            if matched.sum() == rdm.shape[0]:
                count += 1
        else:
            count += matched.sum() / float(rdm.shape[0])

    return count / float(len(rdms))


