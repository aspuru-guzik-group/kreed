import sys
sys.path.append('.')
from src.datamodule import ConformerDatamodule

import torch
qm9_full = ConformerDatamodule(
    dataset="qm9",
    seed=100,
    batch_size=512,
    split_ratio=(0.8, 0.1, 0.1),
    num_workers=8,
    distributed=False,
    tol=-1.0,
    p_drop_labels=0.0,
)

G = qm9_full.datasets['train'][1]
print(G.ndata['cond_labels'])
print(G.ndata['cond_mask'])


qm9_drop = ConformerDatamodule(
    dataset="qm9",
    seed=100,
    batch_size=512,
    split_ratio=(0.8, 0.1, 0.1),
    num_workers=8,
    distributed=False,
    tol=-1.0,
    p_drop_labels=0.1,
)

G = qm9_drop.datasets['train'][1]
print(G.ndata['cond_labels'])
print(G.ndata['cond_mask'])
