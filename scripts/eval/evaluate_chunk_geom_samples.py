import sys
sys.path.append('.')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='')
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--chunk', type=int)
args = parser.parse_args()

directory = args.directory
threshold = args.threshold
k = args.k
chunk = args.chunk
dataset = 'geom'


import dgl
from src.metrics import evaluate_prediction
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
from src import kraitchman, chem
path = Path(directory) / str(chunk)

# all_samples_rebatched is a list of length N, where elements are lists of Molecule objects

import numpy as np

all_sample_coords = np.load(path / 'all_sample_coords.npy')

from src.datamodule import ConformerDatamodule

data = ConformerDatamodule(
    dataset=dataset,
    seed=100,
    batch_size=1,
    split_ratio=(0.8, 0.1, 0.1),
    num_workers=0,
    distributed=False,
    tol=-1.0,
)

dataset = data.datasets['test']
idxs = list(range(len(dataset)))

num_chunks = 1000

def split(a, n, i):
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]

dataset = [dataset[i] for i in split(idxs, num_chunks, chunk)]
N = len(dataset)

# converts a tall stack of coordinates
# into a list of list of coordinates
sizes = np.array([dataset[i].ndata['coords'].shape[0] for i in range(N)])
sizes *= 10
start_idxs = np.cumsum(sizes) - sizes
by_example = np.split(all_sample_coords, start_idxs[1:])
rebatched = []
for sample_block in by_example:
    rebatched.append(np.split(sample_block, 10))


top_1_rmsds = []
top_k_rmsds = []

top_1_correctness = 0
top_1_rmsd_below_threshold = 0

top_k_correctness = 0
top_k_rmsd_below_threshold = 0

n_counted = 0

import numpy as np


pbar = tqdm(zip(rebatched, dataset))
all_results = []
all_aligned_coords = []
for sample_coords, G_true in pbar:
    M_true = chem.Molecule.from_dgl(G_true)
    samples = [M_true.replace(coords=torch.tensor(sample_coords[i])) for i in range(len(sample_coords))]
    n_counted += 1

    results = [evaluate_prediction(M_pred, M_true) for M_pred in samples]
    unsigned_rmsds = [result['unsigned_coords_rmse'] for result in results]

    # sort by substitution coords rmsd
    reorder_idxs = np.argsort(unsigned_rmsds)

    sorted_flipped_coords = []
    sorted_results = []
    for i in reorder_idxs:
        M_pred = samples[i]
        result = results[i]
        transform = result['transform']

        all_aligned_coords.extend(M_pred.transform(transform).coords.numpy())
        sorted_results.append(result)

    all_results.append(sorted_results)

    correctnesses = []
    heavy_rmsds = []
    for result in sorted_results[:k]:
        correctnesses.append(result['correctness'])
        heavy_rmsds.append(result['heavy_coord_rmse'])
    
    top_1_rmsds.append(heavy_rmsds[0])
    top_k_rmsds.append(min(heavy_rmsds))

    top_1_correctness += correctnesses[0]
    top_1_rmsd_below_threshold += float(heavy_rmsds[0] < threshold)

    top_k_correctness += max(correctnesses)
    top_k_rmsd_below_threshold += max([float(rmsd < threshold) for rmsd in heavy_rmsds])

    pbar.set_description(f"Top 1 correctness: {top_1_correctness / n_counted * 100}%")


print(f"Top 1 correctness: {top_1_correctness / N * 100}%")
print(f"Top 1 heavy_rmsd < {threshold}: {top_1_rmsd_below_threshold / N * 100}%")

print(f"Top {k} correctness: {top_k_correctness / N * 100}%")
print(f"Top {k} heavy_rmsd < {threshold}: {top_k_rmsd_below_threshold / N * 100}%")

print("Top 1 median RMSD:", np.median(top_1_rmsds))
print(f"Top {k} median RMSD:", np.median(top_k_rmsds))


with open(path / 'all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

all_aligned_coords = np.array(all_aligned_coords)
np.save(path / 'all_sample_coords.npy', all_aligned_coords)
