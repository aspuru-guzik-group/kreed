import torch
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()
ATOM_MASSES = torch.tensor([0] + [PTABLE.GetAtomicWeight(z) for z in range(1, 119)], dtype=torch.float64)


def rotate_to_principal_axes(atomic_nums, coords):  # (... N) (... N 3)
    masses = ATOM_MASSES[atomic_nums]
    coords = coords.double()

    coords = center_coords(masses, coords)

    I = inertia_dyadic(masses, coords)
    moments, V = torch.linalg.eigh(I)

    return (coords @ V).float()


def center_coords(masses, coords):
    center_of_mass = torch.sum(masses.unsqueeze(-1) * coords, dim=-2) / torch.sum(masses, dim=-1)
    return coords - center_of_mass


def inertia_dyadic(masses, coords):
    m = masses
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]  # (... N)

    x2 = x ** 2.0
    y2 = y ** 2.0
    z2 = z ** 2.0

    Ixx = torch.sum(m * (y2 + z2), dim=-1)
    Iyy = torch.sum(m * (x2 + z2), dim=-1)
    Izz = torch.sum(m * (x2 + y2), dim=-1)
    Ixy = Iyx = -torch.sum(m * x * y, dim=-1)
    Iyz = Izy = -torch.sum(m * y * z, dim=-1)
    Ixz = Izx = -torch.sum(m * x * z, dim=-1)

    return torch.stack([
        torch.stack([Ixx, Ixy, Ixz], dim=-1),
        torch.stack([Iyx, Iyy, Iyz], dim=-1),
        torch.stack([Izx, Izy, Izz], dim=-1)
    ], dim=-2)
