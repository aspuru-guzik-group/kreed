import dgl
import torch


def _mean(G, xyz):
    with G.local_scope():
        G.ndata["tmp"] = xyz
        mean = dgl.mean_nodes(G, "tmp")
    return mean


def zeroed_com(G, xyz):
    return xyz - dgl.broadcast_nodes(G, _mean(G, xyz))


def assert_zeroed_com(G, xyz=None):
    if xyz is None:
        xyz = G.ndata["xyz"]
    com = _mean(G, xyz)
    error = com.abs().max().item() / xyz.abs().max().item()
    assert error < 1e-4, error


def gaussian_KL_div(G, q_mean, q_std, p_mean, p_std, d):
    assert q_std.ndim == p_std.ndim == 1

    with G.local_scope():
        G.ndata["mean_diff"] = (q_mean - p_mean) ** 2.0
        mean_sqe_dist = dgl.sum_nodes(G, "mean_diff").sum(dim=-1)

    kl_div = (2 * d * torch.log(p_std / q_std)) + ((d * q_std.square() + mean_sqe_dist) / p_std.square()) - d
    return 0.5 * kl_div
