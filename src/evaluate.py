import torch

FLIPS = [
    torch.tensor([-1, -1, -1]),
    torch.tensor([-1, -1, 1]),
    torch.tensor([-1, 1, -1]),
    torch.tensor([-1, 1, 1]),
    torch.tensor([1, -1, -1]),
    torch.tensor([1, -1, 1]),
    torch.tensor([1, 1, -1]),
    torch.tensor([1, 1, 1]),
]


def get_greedy_mapping(atom_nums, xyz_pred, xyz_true):
    N = xyz_pred.shape[0]
    mapping = []
    mapped = torch.ones_like(atom_nums)
    BIG = 100
    for i in range(N):
        dists = torch.norm(xyz_pred[i] - xyz_true, dim=1)
        different_type = torch.where(atom_nums[i] != atom_nums, BIG, 1)

        # sort dists and get the index of the closest atom
        idxs_by_dist = torch.argsort(dists * different_type * mapped)
        idx = idxs_by_dist[0]
        mapping.append(idx.item())
        mapped[idx] = BIG
    return mapping


def align_and_rmsd(atom_nums, xyz_pred, xyz_true):
    min_heavy_rmsd = 1000000
    best_flip = FLIPS[-1]

    for flip in FLIPS:
        heavy = (atom_nums != 1)
        heavy_xyz_pred = (xyz_pred * flip)[heavy]
        heavy_xyz_true = xyz_true[heavy]
        heavy_atom_nums = atom_nums[heavy]

        mapping = get_greedy_mapping(heavy_atom_nums, heavy_xyz_pred, heavy_xyz_true)
        heavy_rmsd = (heavy_xyz_pred - heavy_xyz_true[mapping]).square().mean().sqrt()

        if heavy_rmsd < min_heavy_rmsd:
            min_heavy_rmsd = heavy_rmsd
            best_flip = flip

    return min_heavy_rmsd, best_flip


def connectivity_correctness(M_pred, M_true):
    try:
        smiles_pred = M_pred.smiles()
        smiles_true = M_true.smiles()
        return float(smiles_pred == smiles_true)
    except:
        return 0.0


def evaluate(M_pred, M_true):
    metrics = {
        "correctness": connectivity_correctness(M_pred=M_pred, M_true=M_true)
    }

    # atom_nums = M_true.atom_nums
    # cond_mask = M_true.cond_mask
    #
    # heavy = (atom_nums != 1).squeeze(-1)
    # cxyz_pred = M_pred.xyz[heavy][cond_mask[heavy]].abs()
    # cxyz_true = M_true.xyz[heavy][cond_mask[heavy]].abs()
    #
    # abs_C_rmsd = (cxyz_pred - cxyz_true).square().mean().sqrt()
    # heavy_rmsd, best_flip = align_and_rmsd(atom_nums.squeeze(-1), M_pred.xyz, M_true.xyz)

    return metrics
