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


def coord_rmse(M_pred, M_true):
    atom_nums = M_pred.atom_nums  # (N 1)
    coords_pred = kraitchman.rotated_to_principal_axes(M_pred.coords, M_pred.masses, return_moments=False)
    coords_true = kraitchman.rotated_to_principal_axes(M_true.coords, M_true.masses, return_moments=False)

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

    return {"coord_rmse": rmses.min().sqrt().item()}


def cond_rmses(M_pred):
    coords, moments = kraitchman.rotated_to_principal_axes(M_pred.coords, M_pred.masses)

    cond_errors = torch.square(coords.abs() - M_pred.cond_labels)
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
def evaluate(M_pred, M_true):
    return {
        **connectivity_correctness(M_pred=M_pred, M_true=M_true),
        **cond_rmses(M_pred=M_pred),
        **coord_rmse(M_pred=M_pred, M_true=M_true),
    }
