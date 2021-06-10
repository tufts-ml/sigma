import torch
from torch_geometric.utils import to_dense_adj


def reward_general(e1, e2, bc1, bc2, theta, mask1, mask2):
    e1 = to_dense_adj(e1, bc1)
    e2 = to_dense_adj(e2, bc2)
    s_mask = mask1.unsqueeze(-1) & mask1.unsqueeze(-2)
    t_mask = mask2.unsqueeze(-1) & mask2.unsqueeze(-2)
    e1 = e1.masked_fill(~s_mask, 0.0)
    e2 = e2.masked_fill(~t_mask, 0.0)

    e1 = e1.unsqueeze(1)
    e2 = e2.unsqueeze(1)

    r = e1 @ theta @ e2.transpose(-1, -2) * theta
    r = r.sum([1, 2]).mean()

    return r


# Reuse loss in deep graph consensus
# https://openreview.net/pdf?id=HyeJf1HKvS
def dgmc_loss(S, y, reduction='mean'):
    assert reduction in ['none', 'mean', 'sum']
    if not S.is_sparse:
        val = S[:, y[0], y[1]]
    else:
        assert S.__idx__ is not None and S.__val__ is not None
        mask = S.__idx__[y[0]] == y[1].view(-1, 1)
        val = S.__val__[[y[0]]][mask]
    nll = -torch.log(val + 1e-8)
    return nll if reduction == 'none' else getattr(torch, reduction)(nll)


# Reuse hit@1 in deep graph consensus
# https://openreview.net/pdf?id=HyeJf1HKvS
def dgmc_acc(S, y, reduction='mean'):
    assert reduction in ['mean', 'sum']
    if not S.is_sparse:
        pred = S[y[0]].argmax(dim=-1)
    else:
        assert S.__idx__ is not None and S.__val__ is not None
        pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

    correct = (pred == y[1]).sum().item()
    return correct / y.size(1) if reduction == 'mean' else correct
