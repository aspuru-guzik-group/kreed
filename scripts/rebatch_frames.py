import sys
sys.path.append('.')
import dgl
from src.evaluate import evaluate
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
path = Path('v_geom_samples')

with open(path / 'truths.pkl', 'rb') as f:
    truths = pickle.load(f)

with open(path / 'all_frames_batched_by_inference.pkl', 'rb') as f:
    all_frames_batched = pickle.load(f)

with open(path / 'all_reorder_idxs.pkl', 'rb') as f:
    all_reorder_idxs = pickle.load(f)

with open(path / 'reordered_best_flips.pkl', 'rb') as f:
    reordered_best_flips = pickle.load(f)

# create a list of N examples, where each example is a list of 4 sample trajectories, ranked
# only store xyz positions
trajs = [] # (Nexamples, Nsamples, T/render_every, Nnodes, 3)
T = 1000
render_every_n_steps = 5
samples_per_example = 4
steps = [T] + list(reversed(range(-1, T + 1, render_every_n_steps)))

batch_size = 64

def get_idxs(start):
    idxs = [] # store (sample_idx, example_idx) tuples
    for i in range(batch_size):
        idxs.append(((start+i) % samples_per_example, (start+i) // samples_per_example))
    return idxs

for batch_num, inference_batch in tqdm(enumerate(all_frames_batched), total=len(all_frames_batched)):
    for step in tqdm(steps):
        
        batched_frame = inference_batch[step]
        frames = dgl.unbatch(batched_frame)

        for frame, idxs in zip(frames, get_idxs(batch_num*batch_size)):
            sample_idx, example_idx = idxs

            if len(trajs) <= example_idx:
                trajs.append([])
            if len(trajs[example_idx]) <= sample_idx:
                trajs[example_idx].append([])
            
            trajs[example_idx][sample_idx].append(frame.ndata['xyz'].cpu().numpy())

# reorder trajs
new_trajs = []
for example_idx, reorder_idxs in enumerate(all_reorder_idxs):
    new_trajs.append([])
    for reorder_idx in reorder_idxs:
        this_traj = trajs[example_idx][reorder_idx]
        new_trajs[example_idx].append(this_traj)

# apply flips to all trajs
for example_idx, best_flips in enumerate(reordered_best_flips):
    for sample_idx, best_flip in enumerate(best_flips):
        for step in range(len(new_trajs[example_idx][sample_idx])):
            new_trajs[example_idx][sample_idx][step] = new_trajs[example_idx][sample_idx][step] * best_flip.numpy()

# save trajs
with open(path / 'trajs.pkl', 'wb') as f:
    pickle.dump(new_trajs, f)
