
import torch

def get_greedy_mapping(xyz_pred, xyz_truth, atom_nums):
    N = xyz_pred.shape[0]
    mapping = []
    mapped = set()
    for i in range(N):
        dists = torch.norm(xyz_pred[i] - xyz_truth, dim=1)
        # sort dists and get the index of the closest atom
        idxs_by_dist = torch.argsort(dists)
        
        for j in idxs_by_dist:
            j = j.item()
            if j not in mapped and atom_nums[i] == atom_nums[j]:
                mapping.append(j)
                mapped.add(j)
                break

    return mapping

def get_flip(G_pred, G_truth):
    mask = torch.logical_and(G_pred.ndata['cond_mask'], torch.abs(G_pred.ndata['xyz']) > .5)
    mask = torch.all(mask, dim=1)
    nonzero = torch.nonzero(mask)
    if nonzero.shape[0] > 0:
        # there are no labeled atoms far from origin
        carbon_idx = nonzero[0][0]
    else:
        # pick any labeled atom
        carbon_idx = torch.all(G_pred.ndata['cond_mask'], dim=1).nonzero()[0][0]

    flip = G_pred.ndata['xyz'][carbon_idx] / torch.abs(G_truth.ndata['xyz'][carbon_idx])
    flip = flip > 0
    flip = flip * 2 - 1

    return flip

def align_and_rmsd(G_pred, G_truth):
    flip = get_flip(G_pred, G_truth)
    xyz_pred = G_pred.ndata['xyz'] * flip
    xyz_truth = G_truth.ndata['xyz']
    atom_nums = G_pred.ndata['atom_nums']

    # heavy atoms only
    xyz_pred = xyz_pred[atom_nums != 1]
    xyz_truth = xyz_truth[atom_nums != 1]
    atom_nums = atom_nums[atom_nums != 1]
    mapping = get_greedy_mapping(xyz_pred, xyz_truth, atom_nums)
    heavy_rmsd = torch.norm(xyz_pred - xyz_truth[mapping]).mean().sqrt()

    return heavy_rmsd, flip

def evaluate(G_pred, G_truth):
    coords_pred = G_pred.ndata['xyz']
    coords_true = G_truth.ndata['xyz']
    atom_nums = G_pred.ndata['atom_nums']

    carbon_mask = (atom_nums == 6)

    cpred = coords_pred[carbon_mask][G_pred.ndata['cond_mask'][carbon_mask]].abs()
    ctrue = coords_true[carbon_mask][G_truth.ndata['cond_mask'][carbon_mask]].abs()

    abs_C_rmsd = torch.norm(cpred - ctrue).mean().sqrt()

    heavy_rmsd, best_flip = align_and_rmsd(G_pred, G_truth)

    return abs_C_rmsd, heavy_rmsd, best_flip
