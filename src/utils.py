import dgl
import torch


def _mean(G, xyz):
    with G.local_scope():
        G.ndata["tmp"] = xyz
        mean = dgl.mean_nodes(G, "tmp")
    return mean

def _weighted_mean(G, xyz):
    with G.local_scope():
        G.ndata["tmp"] = xyz
        mean = dgl.sum_nodes(G, "tmp", weight="atom_masses") / dgl.sum_nodes(G, "atom_masses")
    return mean

def get_shift(G, xyz):
    with G.local_scope():
        total_masses = dgl.sum_nodes(G, 'atom_masses') # (B 1)
        G.ndata['m'] = G.ndata['atom_masses'] / dgl.broadcast_nodes(G, total_masses) # (N 1)

        G.ndata['m2'] = G.ndata['m'].square() # (N 1)
        m_norm = dgl.sum_nodes(G, 'm2') # (B 1)

        G.ndata['tmpxyz'] = xyz
        pcom = dgl.sum_nodes(G, 'tmpxyz', 'm') # (B 3)
        G.ndata['pcom'] = dgl.broadcast_nodes(G, pcom) # (N 3)

        shift = G.ndata['pcom'] * G.ndata['m'] / dgl.broadcast_nodes(G, m_norm) # (N 3)
        return shift

def orthogonal_projection(G, xyz):
    return xyz - get_shift(G, xyz)

def assert_orthogonal_projection(G, xyz=None):
    if xyz is None:
        xyz = G.ndata["xyz"]
    diff = xyz - orthogonal_projection(G, xyz)
    error = diff.abs().max().item() / (xyz.abs().max().item() + 1e-8)
    assert error < 1e-4, error


def gaussian_KL_div(G, q_mean, q_std, p_mean, p_std, d):
    assert q_std.ndim == p_std.ndim == 1

    with G.local_scope():
        G.ndata["mean_diff"] = (q_mean - p_mean) ** 2.0
        mean_sqe_dist = dgl.sum_nodes(G, "mean_diff").sum(dim=-1)

    kl_div = (2 * d * torch.log(p_std / q_std)) + ((d * q_std.square() + mean_sqe_dist) / p_std.square()) - d
    return 0.5 * kl_div
