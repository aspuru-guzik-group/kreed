import sys
sys.path.append('.')

# parameters:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='final_checkpoints/geom.ckpt')
parser.add_argument('--p_drop', type=float, default=0.0)
parser.add_argument('--samples_per_example', type=int, default=10)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--chunk', type=int)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()

directory = args.directory
checkpoint_path = args.checkpoint_path
dataset = 'geom'
p_drop = args.p_drop
samples_per_example = args.samples_per_example
split = args.split
chunk = args.chunk
threshold = args.threshold
k = args.k

# qm9 test set is 13k examples
# geom test set is 690k examples
from src.diffusion.lit import LitEquivariantDDPM
from src.experimental.train import TrainEquivariantDDPMConfig # necessary
model = LitEquivariantDDPM.load_from_checkpoint(checkpoint_path).to('cuda:0')
import dgl
from src.datamodule import ConformerDatamodule
from tqdm import tqdm
import dgl
import pickle
from torch.utils.data import Dataset, DataLoader
from src import chem
import torch
import numpy as np
from pathlib import Path

#======= SETUP =======#

path = Path(directory) / str(chunk)
path.mkdir(parents=True, exist_ok=True)


data = ConformerDatamodule(
    dataset=dataset,
    seed=100,
    batch_size=1,
    split_ratio=(0.8, 0.1, 0.1),
    num_workers=0,
    distributed=False,
    tol=-1.0,
)

if dataset == 'geom':
    B = 300
elif dataset == 'qm9':
    B = 2500

dataset = data.datasets[split]
idxs = list(range(len(dataset)))

num_chunks = 1000

def split(a, n, i):
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]

dataset = [dataset[i] for i in split(idxs, num_chunks, chunk)]
N = len(dataset)

# generate samples first

examples_to_run = []
for G_true in dataset:
    if p_drop > 0:
        dropout_mask = (torch.rand_like(G_true.ndata['coords']) < p_drop)
        dropout_mask |= (G_true.ndata['atom_nums'] != 6)
        cond_mask = G_true.ndata['cond_mask'] & (~dropout_mask)
        G_true.ndata['cond_mask'] = cond_mask
        G_true.ndata['cond_labels'] = torch.where(cond_mask, G_true.ndata['cond_labels'], 0.0)

    examples_to_run.extend([G_true for _ in range(samples_per_example)])

print(len(examples_to_run), 'examples to run')

# dataloader:
class InferenceDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx].to('cuda:0')

# ======= RUN ======= #

import pickle
progress_path = path / 'progress.pkl'
# check if path exists
if progress_path.exists():
    with open(progress_path, 'rb') as f:
        progress, all_sample_coords, all_cond_masks = pickle.load(f)
else:
    progress = 0
    all_sample_coords = []
    all_cond_masks = []

inference_dataset = InferenceDataset(examples_to_run[progress:])

dgl_collate = dgl.dataloading.GraphCollator().collate
collate_fn = lambda items: chem.Molecule.from_dgl(dgl_collate(items))

loader = DataLoader(
    dataset=inference_dataset,
    batch_size=B,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn,
)


for M in tqdm(loader):

    sample = model.ema.ema_model.sample(M)
    samples = sample.cpu().unbatch()
    for s in samples:
        all_sample_coords.extend(s.coords.numpy())

    all_cond_masks.extend(M.cond_mask.cpu().numpy())

    progress += B
    with open(path / 'progress.pkl', 'wb') as f:
        pickle.dump((progress, all_sample_coords, all_cond_masks), f)

all_sample_coords = np.array(all_sample_coords)
np.save(path / 'all_sample_coords.npy', all_sample_coords)

if p_drop > 0:
    all_cond_masks = np.array(all_cond_masks)
    np.save(path / 'all_cond_masks.npy', all_cond_masks)

# converts a tall stack of coordinates
# into a list of list of coordinates
sizes = np.array([dataset[i].ndata['coords'].shape[0] for i in range(N)])
sizes *= 10
start_idxs = np.cumsum(sizes) - sizes
by_example = np.split(all_sample_coords, start_idxs[1:])
rebatched = []
for sample_block in by_example:
    rebatched.append(np.split(sample_block, 10))


# ======= EVAL ======= #

# now evaluate
top_1_rmsds = []
top_k_rmsds = []

top_1_correctness = 0
top_1_rmsd_below_threshold = 0

top_k_correctness = 0
top_k_rmsd_below_threshold = 0

n_counted = 0

import numpy as np
from src.metrics import evaluate_prediction


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

with open(path / 'eval.txt', 'w') as f:
    f.write(f"Top 1 correctness: {top_1_correctness / N * 100}%")
    f.write(f"Top 1 heavy_rmsd < {threshold}: {top_1_rmsd_below_threshold / N * 100}%")

    f.write(f"Top {k} correctness: {top_k_correctness / N * 100}%")
    f.write(f"Top {k} heavy_rmsd < {threshold}: {top_k_rmsd_below_threshold / N * 100}%")

    f.write(f"Top 1 median RMSD: {np.median(top_1_rmsds)}")
    f.write(f"Top {k} median RMSD: {np.median(top_k_rmsds)}")


with open(path / 'all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

all_aligned_coords = np.array(all_aligned_coords)
np.save(path / 'all_sample_coords.npy', all_aligned_coords)
