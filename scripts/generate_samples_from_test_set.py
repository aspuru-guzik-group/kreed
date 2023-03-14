import sys
sys.path.append('.')
from src.diffusion.lit import LitEquivariantDDPM, LitEquivariantDDPMConfig
from src.experimental.train import TrainEquivariantDDPMConfig
checkpoint_path = 'logs/wandb/latest-run/files/last.ckpt'
model = LitEquivariantDDPM.load_from_checkpoint(checkpoint_path).to('cuda:0')

from src.evaluate import evaluate
import dgl
from src.datamodules import GEOMDatamodule, QM9Datamodule
from src.kraitchman import ATOM_MASSES
from tqdm import tqdm
import dgl
import pickle
from torch.utils.data import Dataset

from pathlib import Path
path = Path('qm9_samples')

dataset = 'qm9'

print('loading datamodule... ', end='')
if dataset == 'geom':
    data = GEOMDatamodule(100, 1)
    B = 64
    samples_per_example = 10
elif dataset == 'qm9':
    data = QM9Datamodule(100, 1)
    samples_per_example = 4
    B = 20_000
    render_every_n_steps = 5
print('done!')


dataset = data.datasets['test']

n_examples = 5000

# pick random numbers between 0 and len(dataset)
import torch
torch.manual_seed(100)
random_indices = torch.randint(0, len(dataset), (n_examples,))

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
