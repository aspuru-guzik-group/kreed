from rdkit.Chem import GetPeriodicTable
import torch
import sys
sys.path.append('../..')
from src import chem, datamodule
import einops
import dgl
import math

PTABLE = GetPeriodicTable()

def constants_to_planar_moments(A,B,C):
    conversion = 5.05376e5
    Ix = conversion / A
    Iy = conversion / B
    Iz = conversion / C

    Px = 0.5 * (-Ix + Iy + Iz)
    Py = 0.5 * (Ix - Iy + Iz)
    Pz = 0.5 * (Ix + Iy - Iz)

    return Px, Py, Pz

def kra_to_molecule(ground_truth, kra, rot):
    # formula_dict = make_formula_dict([t for t, x, y, z in kra])
    if ground_truth is None:
        ground_truth = []
        for t_, x_, y_, z_ in kra:
            ground_truth.append((t_, 0.0, 0.0, 0.0))

    assert len(ground_truth) == len(kra), f"{len(ground_truth)} != {len(kra)}"

    atom_nums = []
    coords = []
    cond_labels = []
    cond_mask = []
    for i in range(len(ground_truth)):
        t, x, y, z = ground_truth[i]
        t_, x_, y_, z_ = kra[i]

        assert t == t_, f"{t} != {t_}"
        atom_nums.append(PTABLE.GetAtomicNumber(t))
        coords.append([x, y, z])
        
        label = []
        mask = []
        for q in [x_, y_, z_]:
            if q is None:
                label.append(0)
                mask.append(0)
            else:
                label.append(abs(q))
                mask.append(1)
        cond_labels.append(label)
        cond_mask.append(mask)

    moments = constants_to_planar_moments(*rot)

    n = len(atom_nums)
    atom_nums = torch.tensor(atom_nums, dtype=torch.long).unsqueeze(-1)
    cond_labels = torch.tensor(cond_labels, dtype=torch.float)
    cond_mask = torch.tensor(cond_mask, dtype=torch.bool)
    masses = chem.atom_masses_from_nums(atom_nums)
    masses_normalized = masses / masses.sum()
    coords = torch.tensor(coords, dtype=torch.float)
    moments = einops.repeat(torch.tensor(moments, dtype=torch.float), "c -> n c", n=coords.shape[0])
    geom_id = torch.full([n, 1], 0, dtype=torch.long)

    edges = datamodule._EDGE_CACHE[:(n * (n - 1)), :]
    u, v = edges[:, 0], edges[:, 1]
    G = dgl.graph((u, v), num_nodes=n)

    # Wrapper
    return chem.Molecule(
        graph=G,
        coords=coords, atom_nums=atom_nums, masses=masses, masses_normalized=masses_normalized,
        cond_labels=cond_labels, cond_mask=cond_mask,
        moments=moments, id=geom_id,
    )

atom_num_to_mass_difference = {
    5: PTABLE.GetMassForIsotope(5, 11) - PTABLE.GetMassForIsotope(5, 10),
    6: PTABLE.GetMassForIsotope(6, 13) - PTABLE.GetMassForIsotope(6, 12),
    7: PTABLE.GetMassForIsotope(7, 15) - PTABLE.GetMassForIsotope(7, 14),
    8: PTABLE.GetMassForIsotope(8, 18) - PTABLE.GetMassForIsotope(8, 16),
    14: PTABLE.GetMassForIsotope(14, 29) - PTABLE.GetMassForIsotope(14, 28),
    16: PTABLE.GetMassForIsotope(16, 34) - PTABLE.GetMassForIsotope(16, 32),
    17: PTABLE.GetMassForIsotope(17, 37) - PTABLE.GetMassForIsotope(17, 35),
    35: PTABLE.GetMassForIsotope(35, 81) - PTABLE.GetMassForIsotope(35, 79),
    80: PTABLE.GetMassForIsotope(80, 202) - PTABLE.GetMassForIsotope(80, 200),
}

def kraitchman(parent_rot, iso_rot):
    atomic_symbols = [x[0] for x in iso_rot]
    mass = sum([PTABLE.GetMostCommonIsotopeMass(PTABLE.GetAtomicNumber(x)) for x in atomic_symbols])
    parent = constants_to_planar_moments(*parent_rot)

    out = []
    for iso in iso_rot:
        at, iso = iso[0], iso[1:]
        if None in iso:
            out.append([at, None, None, None])
            continue
        iso = constants_to_planar_moments(*iso)

        dm = atom_num_to_mass_difference[PTABLE.GetAtomicNumber(at)]
        mu = mass * dm / (mass + dm)

        
        coord = []
        for idx in range(3):

            prod = 1 / mu
            for i in range(3):
                prod *= (iso[i] - parent[idx])

                if i != idx:
                    prod *= 1 / (parent[i] - parent[idx])
        
            coord.append(math.sqrt(prod) if prod > 0 else None)
        out.append([at, *coord])
    return out
