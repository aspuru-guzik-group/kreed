
import torch

def get_greedy_mapping(xyz_pred, xyz_truth, atom_nums):
    N = xyz_pred.shape[0]
    mapping = []
    mapped = torch.ones_like(atom_nums)
    BIG = 100
    for i in range(N):
        dists = torch.norm(xyz_pred[i] - xyz_truth, dim=1)
        different_type = torch.where(atom_nums[i] != atom_nums, BIG, 1)

        # sort dists and get the index of the closest atom
        idxs_by_dist = torch.argsort(dists*different_type*mapped)
        idx = idxs_by_dist[0]
        mapping.append(idx.item())
        mapped[idx] = BIG
    return mapping

flips = [
    torch.tensor([-1, -1, -1]),
    torch.tensor([-1, -1, 1]),
    torch.tensor([-1, 1, -1]),
    torch.tensor([-1, 1, 1]),
    torch.tensor([1, -1, -1]),
    torch.tensor([1, -1, 1]),
    torch.tensor([1, 1, -1]),
    torch.tensor([1, 1, 1]),
]

def align_and_rmsd(G_pred, G_truth):
    min_heavy_rmsd = 1000000
    best_flip = flips[-1]
    for flip in flips:
        xyz_pred = G_pred.ndata['xyz'] * flip
        xyz_truth = G_truth.ndata['xyz']
        atom_nums = G_pred.ndata['atom_nums']

        # heavy atoms only
        xyz_pred = xyz_pred[atom_nums != 1]
        xyz_truth = xyz_truth[atom_nums != 1]
        atom_nums = atom_nums[atom_nums != 1]
        mapping = get_greedy_mapping(xyz_pred, xyz_truth, atom_nums)
        heavy_rmsd = (xyz_pred - xyz_truth[mapping]).square().mean().sqrt()
        if heavy_rmsd < min_heavy_rmsd:
            min_heavy_rmsd = heavy_rmsd
            best_flip = flip

    return min_heavy_rmsd, best_flip

def evaluate(G_pred, G_truth):
    coords_pred = G_pred.ndata['xyz']
    coords_pred = torch.where(torch.isnan(coords_pred), 0.0, coords_pred)
    coords_true = G_truth.ndata['xyz']
    atom_nums = G_pred.ndata['atom_nums']

    nonhydrogen = (atom_nums != 1)

    cpred = coords_pred[nonhydrogen][G_pred.ndata['cond_mask'][nonhydrogen]].abs()
    ctrue = coords_true[nonhydrogen][G_truth.ndata['cond_mask'][nonhydrogen]].abs()

    abs_C_rmsd = (cpred - ctrue).square().mean().sqrt()

    heavy_rmsd, best_flip = align_and_rmsd(G_pred, G_truth)

    correct = connectivity_correct(atom_nums, coords_pred, coords_true)

    return abs_C_rmsd, heavy_rmsd, correct, best_flip

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from src.visualize.html import format_as_xyzfile

def get_smiles(atom_nums, coords):
    xyzfile = format_as_xyzfile(atom_nums, coords)
    mol = Chem.MolFromXYZBlock(xyzfile)
    rdDetermineBonds.DetermineConnectivity(mol)
    return Chem.MolToSmiles(mol)

def connectivity_correct(atom_nums, coords_pred, coords_truth):
    smiles_pred = get_smiles(atom_nums, coords_pred)
    smiles_truth = get_smiles(atom_nums, coords_truth)
    return smiles_pred == smiles_truth
