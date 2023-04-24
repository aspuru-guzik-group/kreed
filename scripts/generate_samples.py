import sys
sys.path.append('.')

# parameters:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='qm9_run_main_samples')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/qm9_run_main/last.ckpt')
parser.add_argument('--dataset', type=str, default='qm9')
parser.add_argument('--p_drop', type=float, default=0.0)
parser.add_argument('--samples_per_example', type=int, default=3)
parser.add_argument('--n_examples', type=int, default=100)
parser.add_argument('--split', type=str, default='test')
args = parser.parse_args()

directory = args.directory
checkpoint_path = args.checkpoint_path
dataset = args.dataset
p_drop = args.p_drop
samples_per_example = args.samples_per_example
n_examples = args.n_examples
split = args.split

# qm9 test set is 13k examples
# geom test set is 69k examples, but you only really need to evaluate on the lowest-energy conformer
from src.diffusion.lit import LitEquivariantDDPM
from src.experimental.train import TrainEquivariantDDPMConfig # necessary
model = LitEquivariantDDPM.load_from_checkpoint(checkpoint_path).to('cuda:0')
import dgl
from src.datamodule import ConformerDatamodule
from tqdm import tqdm
import dgl
import pickle
from torch.utils.data import Dataset

from pathlib import Path
path = Path(directory)
path.mkdir(exist_ok=True)

data = ConformerDatamodule(
    dataset=dataset,
    seed=100,
    batch_size=1,
    split_ratio=(0.8, 0.1, 0.1),
    num_workers=0,
    distributed=False,
    tol=-1.0,
    p_drop_labels=p_drop,
)

render_every_n_steps = 5
if dataset == 'geom':
    B = 200
elif dataset == 'qm9':
    B = 2500

dataset = data.datasets[split]

# pick random numbers between 0 and len(dataset)
import torch
torch.manual_seed(100)
import numpy as np
np.random.seed(102)
if n_examples == -1:
    n_examples = len(dataset)
random_indices = np.random.choice(len(dataset), n_examples, replace=False)

newdataset = []
for idx in random_indices:
    newdataset.append(dataset[idx])

dataset = newdataset
N = len(dataset)

with open(path / 'truths.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# generate samples first

examples_to_run = []
for G_true in dataset:
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

inference_dataset = InferenceDataset(examples_to_run)

loader = dgl.dataloading.GraphDataLoader(inference_dataset, batch_size=B, shuffle=False, drop_last=False, num_workers=0)

T = 1000
keep_frames = set(range(-1, T + 1, render_every_n_steps))

all_samples = []
all_frames = []
for batch in tqdm(loader):
    sample, frames = model.edm.sample_p_G(batch, keep_frames=keep_frames)
    all_samples.append(sample.cpu())
    all_frames.append(frames)

# currently batched according to InferenceDataset, need to rebatch by example
all_samples_flat = []
for sample in all_samples:
    all_samples_flat.extend(dgl.unbatch(sample))

all_samples_rebatched = []
for i in range(0, len(all_samples_flat), samples_per_example):
    all_samples_rebatched.append(dgl.batch(all_samples_flat[i:i+samples_per_example]))

# all_samples_flat is a list of length N * samples_per_example, where elements are dgl graphs
# all_samples_rebatched is a list of length N, where elements are batched dgl graphs

# save all_samples
import pickle
with open(path / 'all_samples_rebatched.pkl', 'wb') as f:
    pickle.dump(all_samples_rebatched, f)

with open(path / 'all_frames_batched_by_inference.pkl', 'wb') as f:
    pickle.dump(all_frames, f)
