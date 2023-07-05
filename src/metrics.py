import itertools

import einops
import scipy
import torch
import numpy as np

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
    # For our purposes, the cost between atoms i and j is their squared distance.
    # However, we have to be careful about not assigning two atoms of different types together.
    # To avoid this, we can set their cost to infinity.
    transformed_costs = torch.square(transformed_coords_preds - coords_true).sum(dim=-1)
    transformed_costs = torch.where(atom_nums == atom_nums.T, transformed_costs, torch.inf)
    transformed_costs = transformed_costs.cpu().numpy()

    # RMSD = root mean squared distance
    # but call it rmse
    rmses = []
    for cost in transformed_costs:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        rmses.append(np.sqrt( np.mean( cost[row_ind, col_ind] ) ))
    rmses = torch.tensor(rmses).to(coords_pred)

    idx = rmses.argmin()
    return rmses[idx].item(), TRANSFORMS[idx]


def connectivity_correctness(M_pred, M_true):
    try:
        smiles_pred = M_pred.smiles()
        smiles_true = M_true.smiles()
        return float(smiles_pred == smiles_true)
    except:
        return 0.0


@torch.no_grad()
def evaluate_prediction(M_pred, M_true, return_aligned_mol=False, keep_coords_pred=False):
    atom_nums = M_pred.atom_nums  # (N 1)

    if keep_coords_pred:
        # Keeping original coordinates because they should already be in the principal axes,
        # but the predicted sample doesn't have CoM = 0. This also means that off-diagonals of
        # inertia matrix are nonzero, and moments_rmse doesn't capture that.
        coords_pred = M_pred.coords
        _, moments_pred = kraitchman.rotated_to_principal_axes(M_pred.coords, M_pred.masses)
    else:
        coords_pred, moments_pred = kraitchman.rotated_to_principal_axes(M_pred.coords, M_pred.masses)
    coords_true = M_true.coords  # assumes M_true is already in principal axes, which it should be if it's from the datamodule

    # Deviation from conditioning information
    cond_errors = torch.square(coords_pred.abs() - M_true.cond_labels)
    cond_errors = cond_errors[M_true.cond_mask]

    n_cond = cond_errors.numel()
    cond_rmse = cond_errors.mean().sqrt().item() if n_cond > 0 else 0.0

    moments_errors = torch.square(moments_pred - M_true.moments[0])
    assert moments_errors.numel() == 3
    moments_rmse = (moments_errors.sum() / 3).sqrt().item()

    # Deviation from inferred molecular graph
    correctness = connectivity_correctness(M_pred=M_pred, M_true=M_true)

    # RMSE on aligned coordinates
    rmse, _ = coord_rmse(
        atom_nums=atom_nums,
        coords_pred=coords_pred,
        coords_true=coords_true,
    )

    # Correctness and RMSE on aligned heavy atom coordinates because hydrogens are not as important
    heavy_mask = (atom_nums.squeeze(-1) != 1)
    heavy_atom_nums = atom_nums[heavy_mask]
    heavy_M_pred = M_pred.replace(atom_nums=heavy_atom_nums, coords=coords_pred[heavy_mask])
    heavy_M_true = M_true.replace(atom_nums=heavy_atom_nums, coords=coords_true[heavy_mask])

    heavy_correctness = connectivity_correctness(M_pred=heavy_M_pred, M_true=heavy_M_true)
    heavy_rmse, transform = coord_rmse(
        atom_nums=heavy_atom_nums,
        coords_pred=heavy_M_pred.coords,
        coords_true=heavy_M_true.coords,
    )

    metrics = {
        "unsigned_coords_rmse": cond_rmse,
        "moments_rmse": moments_rmse,
        "correctness": correctness,
        "heavy_correctness": heavy_correctness,
        "coord_rmse": rmse,
        "heavy_coord_rmse": heavy_rmse,
    }

    aligned_coords = einops.einsum(transform, coords_pred, "i j, n j -> n i")
    M_aligned = M_pred.replace(coords=aligned_coords)
    return (metrics, M_aligned) if return_aligned_mol else metrics
