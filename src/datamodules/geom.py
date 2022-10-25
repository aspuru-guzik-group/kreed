import pathlib

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import src.diffusion.distributions as dists
from src.kraitchman import rotated_to_principal_axes

# GEOM constants
MAX_GEOM_ATOMS = 200
GEOM_ATOMS = torch.tensor([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83], dtype=torch.long)

# Cache to optimize connected graph creation
_MGRID_CACHE = torch.tensor(np.mgrid[0:MAX_GEOM_ATOMS, 0:MAX_GEOM_ATOMS], dtype=torch.long)


class GEOMDataset(Dataset):

    def __init__(self, conformations, tol):
        super().__init__()

        self.conformations = conformations
        self.tol = tol

        self.ztoi = torch.full([84], -100, dtype=torch.long)
        for i, z in enumerate(GEOM_ATOMS):
            self.ztoi[z] = i

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        conformer = self.conformations[idx]

        xyz = conformer["xyz"]
        atom_nums = conformer["atom_nums"]
        n = atom_nums.shape[0]

        # Create a complete graph
        u, v = _MGRID_CACHE[:, 0:n, 0:n]
        u, v = u.flatten(), v.flatten()

        G = dgl.graph((u, v), num_nodes=n)
        G.ndata["atom_nums"] = atom_nums
        G.ndata["xyz"] = xyz

        # Canonicalize conformation
        G = rotated_to_principal_axes(G)

        # Retrieve unsigned coordinates
        abs_xyz = torch.abs(G.ndata["xyz"])

        abs_mask = torch.logical_and(
            (atom_nums == 6),  # carbon
            torch.any(abs_xyz >= self.tol, dim=-1),  # coordinate not too close to axis
        )

        abs_xyz[~abs_mask, :] = 0.0

        G.ndata["abs_xyz"] = abs_xyz
        G.ndata["abs_mask"] = abs_mask

        # Convert atom number to idx
        G.ndata["atom_nums"] = self.ztoi[G.ndata["atom_nums"]]

        # Center molecule coordinates to 0 CoM subspace
        G.ndata["xyz"] = dists.centered_mean(G, G.ndata["xyz"])

        return G


class GEOMDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        seed,
        batch_size=64,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        tol=-1.0,
    ):
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data" / "geom" / "processed"

        # This is a 2D ragged list
        # D[i][j] = j-th conformer for the i-th molecule
        D = torch.load(data_dir / "conformations.pt")

        # Split by molecule
        splits = {"train": None, "val": None, "test": None}
        val_test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        splits["train"], D = train_test_split(D, train_size=split_ratio[0], random_state=seed)
        splits["val"], splits["test"] = train_test_split(D, train_size=val_test_ratio, random_state=(seed + 1))

        datasets = {}
        for n, conformations in splits.items():
            conformations = sum(conformations, [])  # flattens the 2D list
            datasets[n] = GEOMDataset(conformations, tol=tol)
        self.datasets = datasets

    @property
    def d_atom_vocab(self):
        return len(GEOM_ATOMS)

    def train_dataloader(self):
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(split="val", shuffle=False)

    def test_dataloader(self):
        return self._loader(split="test", shuffle=False)

    def _loader(self, split, shuffle, drop_last=False):
        return dgl.dataloading.GraphDataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
        )
