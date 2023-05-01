import sys
sys.path.append('.')

# parameters:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='qm9_run_dev_samples')
parser.add_argument('--checkpoint_path', type=str, default='final_checkpoints/qm9.ckpt')
parser.add_argument('--p_drop', type=float, default=0.0)
parser.add_argument('--samples_per_example', type=int, default=10)
parser.add_argument('--split', type=str, default='test')
args = parser.parse_args()


directory = args.directory
checkpoint_path = args.checkpoint_path
dataset = 'qm9'
p_drop = args.p_drop
samples_per_example = args.samples_per_example
split = args.split

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
path = Path(directory)
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

N = len(dataset)
dataset = [dataset[i] for i in range(N)]

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

dgl_collate = dgl.dataloading.GraphCollator().collate
collate_fn = lambda items: chem.Molecule.from_dgl(dgl_collate(items))

loader = DataLoader(
    dataset=inference_dataset,
    batch_size=B,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    collate_fn=collate_fn,
    # pin_memory=True,
)

all_samples = []
for M in tqdm(loader):
    if p_drop > 0:
        dropout_mask = (torch.rand_like(M.coords) < p_drop)
        cond_mask = M.cond_mask & (~dropout_mask)
        M = M.replace(cond_mask=cond_mask, cond_labels=torch.where(cond_mask, M.cond_labels, 0.0))

    sample = model.edm.sample(M)
    all_samples.extend(sample.cpu().unbatch())

all_samples_rebatched = []
for i in range(0, len(all_samples), samples_per_example):
    items = all_samples[i:i+samples_per_example]
    all_samples_rebatched.append(items)

# all_samples_rebatched is a list of length N, where elements are lists of Molecule objects

# save all_samples
import pickle
with open(path / 'all_samples_rebatched.pkl', 'wb') as f:
    pickle.dump(all_samples_rebatched, f)
