import torch
import torch.linalg as LA


def rotated_to_principal_axes(coords, masses, return_moments=True):
    coords, m = coords.double(), masses.double()

    # Subtract CoM
    com = torch.sum(m * coords, dim=0) / m.sum()
    coords = coords - com

    # Compute planar dyadic
    dyadic = m.unsqueeze(-1) * coords.unsqueeze(-1) * coords.unsqueeze(-2)  # (N 1 1) * (N 3 1) * (N 1 3)
    dyadic = dyadic.sum(dim=0)  # (3 3)

    # Diagonalize in double precision
    moments, V = LA.eigh(dyadic)  # (3) (3 3)

    # Sort eigenvalues in descending order
    moments = torch.flip(moments, dims=[-1])
    V = torch.flip(V, dims=[-1])

    # Sanity check
    Q = V @ torch.diag_embed(moments) @ V.mT
    err = (dyadic - Q).abs().max()
    assert err <= 1e-5, err.item()

    coords = (coords @ V).float()
    return (coords, moments.float()) if return_moments else coords
