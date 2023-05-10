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
args = parser.parse_args()

directory = args.directory
checkpoint_path = args.checkpoint_path
dataset = 'geom'
p_drop = args.p_drop
samples_per_example = args.samples_per_example
split = args.split
chunk = args.chunk

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

from pathlib import Path
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
    B = 200
elif dataset == 'qm9':
    B = 2500

dataset = data.datasets[split]
idxs = list(range(len(dataset)))

num_chunks = 1000

def split(a, n, i):
    k, m = divmod(len(a), n)
    return a[i*k+min(i, m):(i+1)*k+min(i+1, m)]

dataset = [dataset[i].to('cuda:0') for i in split(idxs, num_chunks, chunk)]
N = len(dataset)

# generate samples first

all_cond_masks = []
examples_to_run = []
for G_true in dataset:
    if p_drop > 0:
        dropout_mask = (torch.rand_like(G_true.ndata['coords']) < p_drop)
        cond_mask = G_true.ndata['cond_mask'] & (~dropout_mask)
        G_true.ndata['cond_mask'] = cond_mask
        G_true.ndata['cond_labels'] = torch.where(cond_mask, G_true.ndata['cond_labels'], 0.0)
        all_cond_masks.extend(G_true.ndata['cond_mask'].cpu().numpy())

    examples_to_run.extend([G_true for _ in range(samples_per_example)])

print(len(examples_to_run), 'examples to run')

# dataloader:
class InferenceDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

inference_dataset = InferenceDataset(examples_to_run)

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


all_sample_coords = []
for M in tqdm(loader):

    sample = model.ema.ema_model.sample(M)
    samples = sample.cpu().unbatch()
    for s in samples:
        all_sample_coords.extend(s.coords.numpy())

import numpy as np
all_sample_coords = np.array(all_sample_coords)
np.save(path / 'all_sample_coords.npy', all_sample_coords)

if p_drop > 0:
    all_cond_masks = np.array(all_cond_masks)
    np.save(path / 'all_cond_masks.npy', all_cond_masks)
