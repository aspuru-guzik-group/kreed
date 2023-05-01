import itertools

import einops
import scipy
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


def cond_rmses(M_pred, moments):
    # assume M_pred has been diagonalized already

    cond_errors = torch.square(M_pred.coords.abs() - M_pred.cond_labels)
    cond_errors = torch.where(M_pred.cond_mask, cond_errors, 0.0)
    moments_errors = torch.square(moments - M_pred.moments)

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


@torch.no_grad()
def evaluate_prediction(M_pred, M_true):
    # assume M_true has been diagonalized already
    atom_nums = M_pred.atom_nums  # (N 1)
    coords_pred, moments_pred = kraitchman.rotated_to_principal_axes(M_pred.coords, M_pred.masses)
    # coords_true, moments_true = kraitchman.rotated_to_principal_axes(M_true.coords, M_true.masses)
    coords_true = M_true.coords

    cond_errors = torch.square(coords_pred.abs() - M_true.cond_labels)
    cond_errors = torch.where(M_pred.cond_mask, cond_errors, 0.0)
    moments_errors = torch.square(moments_pred - M_true.moments[0])

    out = coord_rmse(atom_nums=atom_nums, coords_pred=coords_pred, coords_true=coords_true)

    mask = atom_nums.squeeze(-1) != 1
    atom_nums_no_H = atom_nums[mask]
    coords_pred_no_H = coords_pred[mask]
    coords_true_no_H = coords_true[mask]

    # the alignment is done only on the heavy atoms
    # because the hydrogens are not as important
    out_no_H = coord_rmse(atom_nums_no_H, coords_pred_no_H, coords_true_no_H)
    
    # aligned_coords_pred = coords_pred @ out_no_H['transform']
    return {
        "unsigned_coords_rmse": cond_errors.sum().sqrt().item(),
        "moments_rmse": moments_errors.sum().sqrt().item(),
        'coord_rmse': out['coord_rmse'],
        'heavy_coord_rmse': out_no_H['coord_rmse'],
        'transform': out_no_H['transform'],
        **connectivity_correctness(M_pred=M_pred, M_true=M_true),
    }
