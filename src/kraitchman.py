import dgl
import torch
import torch.linalg as LA
from rdkit.Chem.rdchem import GetPeriodicTable

from src.eigh import StablePowerIteration

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)], dtype=torch.float32)


def rotated_to_principal_axes(G, stable=False, return_moments=False, n_iter=19):
    with G.local_scope():
        m = ATOM_MASSES[G.ndata["atom_nums"]]

        # Center molecule to center of mass (CoM)
        G.ndata["m"] = m.unsqueeze(-1)
        coms = dgl.sum_nodes(G, "xyz", weight="m") / dgl.sum_nodes(G, "m")

        # Compute planar matrix of inertia
        xyz = G.ndata["xyz"] - dgl.broadcast_nodes(G, coms)
        xyz = xyz.double()

        # P = sum_{i = 1}^n m_i * r_i * r_i^T
        B, N = G.batch_size, xyz.shape[0]
        P = m.view(N, 1, 1) * torch.bmm(xyz.unsqueeze(-1), xyz.unsqueeze(-2))

        G.ndata["P"] = P.view(N, 9)  # flatten to use dgl.sum_nodes()
        P = dgl.sum_nodes(G, "P").view(B, 3, 3)

    moments, V = LA.eigh(P)  # (B 3) (B 3 3)
    moments = moments.detach()

    # Sort eigenvalues in descending order
    moments = torch.flip(moments, dims=[-1])
    V = torch.flip(V, dims=[-1])

    Q = V @ torch.diag_embed(moments) @ V.transpose(-1, -2)
    err = (P - Q).abs().max()
    assert err <= 1e-10, err.item()

    if stable:
        V = V.detach()

        M = P
        V_diffable = []

        # Numerically stable eigh()
        for i in range(3):
            v = V[:, i].unsqueeze(-1)
            v = StablePowerIteration.apply(M, v, n_iter)
            M = M - torch.bmm(torch.bmm(M, v), v.transpose(-1, -2))
            V_diffable.append(v)

        V = torch.cat(V_diffable, dim=-1)

    V_T = dgl.broadcast_nodes(G, V.transpose(-1, -2))

    G_canon = G.local_var()
    G_canon.ndata["xyz"] = torch.bmm(V_T, xyz.unsqueeze(-1)).squeeze(-1).float()
    return (G_canon, moments) if return_moments else G_canon
