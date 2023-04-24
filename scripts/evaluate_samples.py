import sys
sys.path.append('.')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='qm9_run_main_samples')
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--k', type=int, default=3)
args = parser.parse_args()

directory = args.directory
threshold = args.threshold
k = args.k

import dgl
from src.new_evaluate import evaluate
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
from src import kraitchman
path = Path(directory)

# all_samples_flat is a list of length N * samples_per_example, where elements are dgl graphs
# all_samples_rebatched is a list of length N, where elements are batched dgl graphs

# save all_samples
import pickle

with open(path / 'truths.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open(path / 'all_samples_rebatched.pkl', 'rb') as f:
    all_samples_rebatched = pickle.load(f)

N = len(dataset)

top_1_correctness = 0
top_1_rmsd_below_threshold = 0

top_k_correctness = 0
top_k_rmsd_below_threshold = 0

n_counted = 0

import numpy as np


pbar = tqdm(zip(all_samples_rebatched, dataset))
all_samples = []
all_reorder_idxs = [] # (Nexamples, Nsamples)
all_results = []
for sample, G_true in pbar:
    n_counted += 1
    samples = dgl.unbatch(sample)

    results = [evaluate(G_pred, G_true) for G_pred in samples]
    unsigned_rmsds = [result['unsigned_coords_rmse'] for result in results]

    # sort by substitution coords rmsd
    reorder_idxs = np.argsort(unsigned_rmsds)
    all_reorder_idxs.append(reorder_idxs)

    sorted_flipped_samples = []
    sorted_results = []
    for i in reorder_idxs:
        sample = samples[i]
        result = results[i]
        transform = result['transform']
        new_sample = sample.clone()
        new_sample.ndata['xyz'] = kraitchman.rotated_to_principal_axes(new_sample, False).ndata['xyz'] @ transform

        sorted_flipped_samples.append(new_sample)
        sorted_results.append(result)

    all_results.append(sorted_results)
    all_samples.append(sorted_flipped_samples)

    correctnesses = []
    heavy_rmsds = []
    for result in sorted_results[:k]:
        correctnesses.append(result['correctness'])
        heavy_rmsds.append(result['heavy_coord_rmse'])

    top_1_correctness += correctnesses[0]
    top_1_rmsd_below_threshold += float(heavy_rmsds[0] < threshold)

    top_k_correctness += max(correctnesses)
    top_k_rmsd_below_threshold += max([float(rmsd < threshold) for rmsd in heavy_rmsds])

    pbar.set_description(f"Top 1 correctness: {top_1_correctness / n_counted * 100}%")


print(f"Top 1 correctness: {top_1_correctness / N * 100}%")
print(f"Top 1 heavy_rmsd < {threshold}: {top_1_rmsd_below_threshold / N * 100}%")

print(f"Top {k} correctness: {top_k_correctness / N * 100}%")
print(f"Top {k} heavy_rmsd < {threshold}: {top_k_rmsd_below_threshold / N * 100}%")

with open(path / 'reordered_flipped_samples.pkl', 'wb') as f:
    pickle.dump(all_samples, f)

with open(path / 'all_reorder_idxs.pkl', 'wb') as f:
    pickle.dump(all_reorder_idxs, f) # (Nexamples, Nsamples)

with open(path / 'all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
