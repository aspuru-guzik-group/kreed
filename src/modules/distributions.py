import dgl
import torch


def _mean(G, xyz):
    with G.local_scope():
        G.ndata["tmp"] = xyz
        mean = dgl.mean_nodes(G, "tmp")
    return mean


def centered_mean(G, xyz):
    return xyz - dgl.broadcast_nodes(G, _mean(G, xyz))


def assert_centered_mean(G, xyz):
    com = _mean(G, xyz)
    error = com.abs().max().item() / xyz.abs().max().item()
    assert error < 1e-4, error


def subspace_gaussian_KL_div(G, q_mean, q_var, p_mean, p_var):
    """Computes KL[q(z)||p(z)], where q and p are isotropic Gaussians.
    """

    assert q_var.ndim == p_var.ndim == 1

    with G.local_scope():
        G.ndata["mean_diff"] = (q_mean - p_mean) ** 2.0
        mean_sqe_dist = dgl.sum_nodes(G, "mean_diff").sum(dim=-1)

    d = (G.batch_num_nodes() - 1) * 3
    kl_div = d * torch.log(p_var / q_var) - d + ((mean_sqe_dist + d * q_var) / p_var)
    return 0.5 * kl_div


def gaussian_KL_div(G, q_mean, q_var, p_mean, p_var):
    """Computes KL[q(z)||p(z)], where q and p are isotropic Gaussians. Does not use the mean-centered subspace.
    """

    assert q_var.ndim == p_var.ndim == 1

    with G.local_scope():
        G.ndata["mean_diff"] = torch.where(G.ndata['free_mask'], (q_mean - p_mean) ** 2.0, 0.0)
        mean_sqe_dist = dgl.sum_nodes(G, "mean_diff").sum(dim=-1)

    d = G.ndata['free_mask'].sum()
    kl_div = d * torch.log(p_var / q_var) - d + ((mean_sqe_dist + d * q_var) / p_var)
    # check this later

    return 0.5 * kl_div

def KL_div(q, p):
    """Computes discrete KL divergence KL[ q || p ]
    """
    return (q * (q / p).log()).sum()

