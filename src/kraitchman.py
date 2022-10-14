import dgl
import torch
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetAtomicWeight(z) for z in range(1, 119)], dtype=torch.float64)


def rotate_to_principal_axes(G):
    G.ndata["m"] = ATOM_MASSES[G.ndata["atom_nums"]].unsqueeze(-1)
    G.ndata["xyz"] = G.ndata["xyz"].double()  # increase precision

    # center molecule to center of mass (CoM)
    coms = dgl.sum_nodes(G, "xyz", weight="m") / dgl.sum_nodes(G, "m")
    G.ndata["xyz"] -= dgl.broadcast_nodes(G, coms)

    # compute inertia dyadic matrix
    I = inertia_dyadic(G)

    moments, V = torch.linalg.eigh(I)
    V = dgl.broadcast_nodes(G, V)

    G.ndata["xyz"] = (V @ G.ndata["xyz"].unsqueeze(-1)).squeeze(-1).float()
    G.ndata.pop("m")  # cleanup


def inertia_dyadic(G):
    with G.local_scope():
        xyz = G.ndata["xyz"]

        m = G.ndata["m"].squeeze(-1)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        x2 = x ** 2.0
        y2 = y ** 2.0
        z2 = z ** 2.0

        G.ndata["Ixx"] = m * (y2 + z2)
        G.ndata["Iyy"] = m * (x2 + z2)
        G.ndata["Izz"] = m * (x2 + y2)
        G.ndata["Ixy"] = - m * x * y
        G.ndata["Iyz"] = - m * y * z
        G.ndata["Ixz"] = - m * x * z

        Ixx = dgl.sum_nodes(G, "Ixx")
        Iyy = dgl.sum_nodes(G, "Iyy")
        Izz = dgl.sum_nodes(G, "Izz")
        Ixy = Iyx = dgl.sum_nodes(G, "Ixy")
        Iyz = Izy = dgl.sum_nodes(G, "Iyz")
        Ixz = Izx = dgl.sum_nodes(G, "Ixz")

    return torch.stack([
        torch.stack([Ixx, Ixy, Ixz], dim=-1),
        torch.stack([Iyx, Iyy, Iyz], dim=-1),
        torch.stack([Izx, Izy, Izz], dim=-1)
    ], dim=-2)
