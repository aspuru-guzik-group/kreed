import dgl
import torch
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetMostCommonIsotopeMass(z) for z in range(1, 119)], dtype=torch.float32)


def rotated_to_principal_axes(G, stable=False, return_moments=False):
    with G.local_scope():
        m = ATOM_MASSES[G.ndata["atom_nums"]]

        # Center molecule to center of mass (CoM)
        G.ndata["m"] = m.unsqueeze(-1)
        coms = dgl.sum_nodes(G, "xyz", weight="m") / dgl.sum_nodes(G, "m")

        # Compute planar matrix of inertia
        xyz = G.ndata["xyz"] - dgl.broadcast_nodes(G, coms)

        # P = sum_{i = 1}^n m_i * r_i * r_i^T
        B, N = G.batch_size, xyz.shape[0]
        P = m.view(N, 1, 1) * torch.bmm(xyz.unsqueeze(-1), xyz.unsqueeze(-2))

        G.ndata["P"] = P.view(N, 9)  # flatten to use dgl.sum_nodes()
        P = dgl.sum_nodes(G, "P").view(B, 3, 3)

    if stable:
        raise NotImplementedError()
    else:
        P = P.double()
        moments, V = torch.linalg.eigh(P)  # diagonalize in double precision
        moments, V = moments.float(), V.float()

    V_T = dgl.broadcast_nodes(G, V.transpose(-1, -2))

    G_canon = G.local_var()
    G_canon.ndata["xyz"] = torch.bmm(V_T, xyz.unsqueeze(-1)).squeeze(-1)
    return (G_canon, moments) if return_moments else G_canon
