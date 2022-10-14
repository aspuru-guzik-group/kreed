import numpy as np
from rdkit.Chem.rdchem import GetPeriodicTable

PTABLE = GetPeriodicTable()


def rotate_to_principal_axes(atomic_nums, coords):
    masses = np.array([PTABLE.GetAtomicWeight(z.item()) for z in atomic_nums], dtype=np.float64)
    coords = coords.astype(np.float64)

    coords = center_coords(masses, coords)

    I = inertia_dyadic(masses, coords)
    moments, V = np.linalg.eigh(I)

    return (coords @ V).astype(np.float32)


def center_coords(masses, coords):
    center_of_mass = np.sum(masses[:, None] * coords, axis=0) / np.sum(masses)
    return coords - center_of_mass


def inertia_dyadic(masses, coords):
    m = masses
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    x2 = x ** 2.0
    y2 = y ** 2.0
    z2 = z ** 2.0

    Ixx = np.sum(m * (y2 + z2))
    Iyy = np.sum(m * (x2 + z2))
    Izz = np.sum(m * (x2 + y2))
    Ixy = Iyx = -np.sum(m * x * y)
    Iyz = Izy = -np.sum(m * y * z)
    Ixz = Izx = -np.sum(m * x * z)

    return np.array([
        [Ixx, Ixy, Ixz],
        [Iyx, Iyy, Iyz],
        [Izx, Izy, Izz],
    ])
