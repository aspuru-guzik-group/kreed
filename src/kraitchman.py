import dgl
import torch
import torch.linalg as LA

def rotated_to_principal_axes(G, stable=False, return_moments=False, n_iter=19):
    with G.local_scope():

        # Center molecule to center of mass (CoM)
        coms = dgl.sum_nodes(G, "xyz", weight="atom_masses") / dgl.sum_nodes(G, "atom_masses")

        # Compute planar matrix of inertia
        xyz = G.ndata["xyz"] - dgl.broadcast_nodes(G, coms)
        xyz = xyz.double()

        # P = sum_{i = 1}^n m_i * r_i * r_i^T
        B, N = G.batch_size, xyz.shape[0]
        P = G.ndata['atom_masses'].view(N, 1, 1) * torch.bmm(xyz.unsqueeze(-1), xyz.unsqueeze(-2))

        G.ndata["P"] = P.view(N, 9)  # flatten to use dgl.sum_nodes()
        P = dgl.sum_nodes(G, "P").view(B, 3, 3)

    moments, V = LA.eigh(P)  # (B 3) (B 3 3)
    moments = moments.detach()

    # Sort eigenvalues in descending order
    moments = torch.flip(moments, dims=[-1])
    V = torch.flip(V, dims=[-1])

    V_T = dgl.broadcast_nodes(G, V.transpose(-1, -2))

    G_canon = G.local_var()
    G_canon.ndata["xyz"] = torch.bmm(V_T, xyz.unsqueeze(-1)).squeeze(-1).float()
    return (G_canon, moments) if return_moments else G_canon
