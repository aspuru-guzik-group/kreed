import itertools

import einops
import scipy.optimize
import torch
import torch.nn.functional as F

from src import kraitchman


# ================================================================================================ #
#                                              Caches                                              #
# ================================================================================================ #


def _build_transforms_cache():
    transforms = []
    for flips in itertools.product([-1, 1], repeat=3):
        transforms.append(torch.diag_embed(torch.tensor(flips, dtype=torch.float)))
    return torch.stack(transforms, dim=0)


TRANSFORMS = _build_transforms_cache()


# ================================================================================================ #
#                                             Metrics                                              #
# ================================================================================================ #


def coord_rmse(atom_nums, coords_pred, coords_true):

    transformed_coords_preds = einops.einsum(
        TRANSFORMS.to(coords_pred),
        coords_pred,
        "t i j, n j -> t n i"
    ).unsqueeze(-2)  # (T N 1 3)

    # An T x N x N matrix where transformed_costs[t][i][j] is the cost of assigning atom #i in
    # coords_pred (under the t-th transformation) to atom #j in coords_true.
    # For our purposes, the cost is the MSE between the coordinates of the two atoms.
    # However, we have to be careful about not assigning two atoms of different types together.
    # To avoid this, we can set their cost to infinity.
    transformed_costs = torch.square(transformed_coords_preds - coords_true).sum(dim=-1)
    transformed_costs = torch.where(atom_nums == atom_nums.T, transformed_costs, torch.inf)
    transformed_costs = transformed_costs.cpu().numpy()

    rmses = []
    for cost in transformed_costs:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        rmses.append(cost[row_ind, col_ind].sum())
    rmses = torch.tensor(rmses).to(coords_pred)

    idx = rmses.argmin()
    return {"coord_rmse": rmses[idx].sqrt().item(),
            "transform": TRANSFORMS[idx],
            }


def cond_rmses(G_pred, moments):
    cond_errors = torch.square(G_pred.ndata['xyz'].abs() - G_pred.ndata['cond_labels'])
    cond_errors = torch.where(G_pred.ndata['cond_mask'], cond_errors, 0.0)
    moments_errors = torch.square(moments - G_pred.ndata['moments'][0])

    return {
        "unsigned_coords_rmse": cond_errors.sum().sqrt().item(),
        "moments_rmse": moments_errors.sum().sqrt().item(),
    }


def connectivity_correctness(M_pred, M_true):
    try:
        smiles_pred = M_pred.smiles()
        smiles_true = M_true.smiles()
        result = float(smiles_pred == smiles_true)
    except:
        result = 0.0
    return {"correctness": result}

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from src.visualize.html import format_as_xyzfile

def get_smiles(atom_nums, coords):
    xyzfile = format_as_xyzfile(atom_nums, coords)
    mol = Chem.MolFromXYZBlock(xyzfile)
    rdDetermineBonds.DetermineConnectivity(mol)
    return Chem.MolToSmiles(mol)

def connectivity_correctness(atom_nums, coords_pred, coords_true):

    try:
        smiles_pred = get_smiles(atom_nums, coords_pred)
        smiles_true = get_smiles(atom_nums, coords_true)
        result = float(smiles_pred == smiles_true)
    except:
        result = 0.0
    return {"correctness": result}


@torch.no_grad()
def evaluate(G_pred, G_true):
    with G_pred.local_scope():
        atom_nums = G_pred.ndata['atom_nums']  # (N 1)
        diag_G_pred, moments = kraitchman.rotated_to_principal_axes(G_pred, return_moments=True)
        coords_pred = diag_G_pred.ndata['xyz']
        coords_true = G_true.ndata['xyz']

    out = coord_rmse(atom_nums, coords_pred, coords_true)

    mask = atom_nums.squeeze(-1) != 1
    atom_nums_no_H = atom_nums[mask]
    coords_pred_no_H = coords_pred[mask]
    coords_true_no_H = coords_true[mask]

    # the alignment is done only on the heavy atoms
    # because the hydrogens are not as important
    out_no_H = coord_rmse(atom_nums_no_H, coords_pred_no_H, coords_true_no_H)

    # aligned_coords_pred = coords_pred @ out_no_H['transform']
    return {
        'with_H_coord_rmse': out['coord_rmse'],
        'heavy_coord_rmse': out_no_H['coord_rmse'],
        'transform': out_no_H['transform'],
        **cond_rmses(G_pred, moments),
        **connectivity_correctness(atom_nums, coords_pred, coords_true),
    }
