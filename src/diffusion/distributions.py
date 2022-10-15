import torch
import dgl


def gaussian_KL_divergence(G, q_mean, q_var, p_mean, p_var):
    """Computes KL[q(z)||p(z)], where q and p are isotropic Gaussians.
    """
    
    assert q_var.ndim == p_var.ndim == 1

    with G.local_scope():
        G.ndata["mean_diff"] = (q_mean - p_mean) ** 2.0
        mean_sqe_dist = dgl.sum_nodes(G, "mean_diff").sum(dim=-1)

    d = G.batch_num_nodes() * 3
    kl_div = d * torch.log(p_var / q_var) - d + ((mean_sqe_dist + d * q_var) / p_var)
    return 0.5 * kl_div


