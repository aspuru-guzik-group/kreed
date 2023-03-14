import sys
sys.path.append('.')
import dgl
from src.evaluate import evaluate
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
path = Path('qm9_samples')

# all_samples_flat is a list of length N * samples_per_example, where elements are dgl graphs
# all_samples_rebatched is a list of length N, where elements are batched dgl graphs

# save all_samples
import pickle

with open(path / 'truths.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open(path / 'all_samples_rebatched.pkl', 'rb') as f:
    all_samples_rebatched = pickle.load(f)

N = len(dataset)

threshold = 0.02
top_1_correctness = 0
top_1_rmsd_below_threshold = 0

top_5_correctness = 0
top_5_rmsd_below_threshold = 0

n_counted = 0

def abs_c_rmsd(G_pred, G_true):
    coords_pred = G_pred.ndata['xyz'].cpu()
    coords_true = G_true.ndata['xyz'].cpu()
    atom_nums = G_true.ndata['atom_nums'].cpu()

    nonhydrogen = (atom_nums != 1)

    cpred = coords_pred[nonhydrogen][G_pred.ndata['cond_mask'][nonhydrogen]].abs()
    ctrue = coords_true[nonhydrogen][G_true.ndata['cond_mask'][nonhydrogen]].abs()

    abs_C_rmsd = (cpred - ctrue).square().mean().sqrt()
    return abs_C_rmsd

pbar = tqdm(zip(all_samples_rebatched, dataset))
all_best_flips = []
reordered_samples = []
for sample, G_true in pbar:
    n_counted += 1
    samples = dgl.unbatch(sample)

    abcrmsds = [abs_c_rmsd(G_pred, G_true) for G_pred in samples]

    atom_nums = G_true.ndata['atom_nums'].cpu().numpy()
    coords_true = G_true.ndata['xyz'].cpu().numpy()

    samples, abcrmsds = zip(*sorted(zip(samples, abcrmsds), key=lambda x: x[1]))
    reordered_samples.append(samples)

    results = []
    for G_pred in samples[:5]:
        result = evaluate(G_pred, G_true)
        results.append(result)
    
    abs_C_rmsds, heavy_rmsds, correctnesses, best_flips = zip(*results)
    all_best_flips.append(best_flips)

    top_1_correctness += 1 if correctnesses[0] else 0
    top_1_rmsd_below_threshold += 1 if heavy_rmsds[0] < threshold else 0

    top_5_correctness += 1 if any(correctnesses) else 0
    top_5_rmsd_below_threshold += 1 if any([rmsd < threshold for rmsd in heavy_rmsds]) else 0

    pbar.set_description(f"Top 1 correctness: {top_1_correctness / n_counted}")


print(f"Top 1 correctness: {top_1_correctness / N}")
print(f"Top 1 heavy_rmsd < {threshold}: {top_1_rmsd_below_threshold / N}")

print(f"Top 5 correctness: {top_5_correctness / N}")
print(f"Top 5 heavy_rmsd < {threshold}: {top_5_rmsd_below_threshold / N}")

with open(path / 'all_best_flips.pkl', 'wb') as f:
    pickle.dump(all_best_flips, f)

with open(path / 'reordered_samples.pkl', 'wb') as f:
    pickle.dump(reordered_samples, f)
