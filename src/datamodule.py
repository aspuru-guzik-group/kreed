import pathlib

import dgl
import einops
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import pl_worker_init_function
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src import chem, kraitchman


# ================================================================================================ #
#                                              Caches                                              #
# ================================================================================================ #


def _build_edge_cache(max_nodes):
    cache = []
    for i in range(max_nodes):
        for j in range(i):
            cache.append([i, j])
            cache.append([j, i])
    return torch.tensor(cache, dtype=torch.long)


_EDGE_CACHE = _build_edge_cache(max_nodes=200)


# ================================================================================================ #
#                                         Data Handling                                            #
# ================================================================================================ #


class ConformerDataset(Dataset):

    def __init__(self, conformations, tol):
        super().__init__()

        self.conformations = conformations
        self.tol = tol

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        txyz, geom_id = self.conformations[idx]

        n = txyz.shape[0]
        coords = torch.tensor(txyz[:, 1:], dtype=torch.float)  # (N 3)
        atom_nums = torch.tensor(txyz[:, :1], dtype=torch.long)  # (N 1)

        # Precompute
        masses = chem.atom_masses_from_nums(atom_nums)
        masses_normalized = masses / masses.sum()

        # Canonicalize conformation
        coords, moments = kraitchman.rotated_to_principal_axes(coords, masses)

        # Retrieve unsigned coordinates for isotopically abundant atoms that
        # are not too close to coordinate axis.
        # cond_mask is True at elements of cond_labels that are specified
        cond_mask = torch.isin(atom_nums, chem.ISOTOPICALLY_ABUNDANT_ATOMS) & (coords.abs() >= self.tol)
        cond_labels = torch.where(cond_mask, coords.abs(), 0.0)

        # Create a complete graph
        edges = _EDGE_CACHE[:(n * (n - 1)), :]
        u, v = edges[:, 0], edges[:, 1]
        G = dgl.graph((u, v), num_nodes=n)

        # Store graph-label data as repeated node features
        moments = einops.repeat(moments, "c -> n c", n=n)
        geom_id = torch.full([n, 1], geom_id, dtype=torch.long)

        # Wrapper
        M = chem.Molecule(
            graph=G,
            coords=coords, atom_nums=atom_nums, masses=masses, masses_normalized=masses_normalized,
            cond_labels=cond_labels, cond_mask=cond_mask,
            moments=moments, id=geom_id,
        )

        # Convert to DGL to take advantage of DGL batching
        return chem.Molecule.to_dgl(M)


class ConformerDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        dataset,
        seed,
        batch_size,
        split_ratio,
        tol,
        num_workers=0,
        distributed=False,
    ):
        super().__init__()

        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.tol = tol

        # Load data
        data_dir = pathlib.Path(__file__).parents[1] / "data" / dataset / "processed"
        metadata = np.load(str(data_dir / "metadata.npy"))
        coords = np.load(str(data_dir / "coords.npy"))

        # Unbind coordinates
        start_indices = metadata[:, 0]
        coords = np.split(coords, start_indices[1:])

        # Group conformations by molecule
        D = dict()
        for info, txyz in zip(metadata, coords):
            _, molecule_id, geom_id = list(info)
            if molecule_id not in D:
                D[molecule_id] = list()
            D[molecule_id].append((txyz, geom_id))
        D = list(D.values())
        print(f"Dataset: {sum(len(C) for C in D)} conformations from {len(D)} molecules.")

        # Create train/val/test split
        splits = {"train": None, "val": None, "test": None}
        val_test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        splits["train"], D = train_test_split(D, train_size=split_ratio[0], random_state=seed)
        splits["val"], splits["test"] = train_test_split(D, train_size=val_test_ratio, random_state=seed + 1)

        # Create PyTorch datasets
        datasets = {}
        for split, D_split in splits.items():
            conformations = []
            for mol_conformers in D_split:
                conformations.extend(mol_conformers)
            datasets[split] = ConformerDataset(conformations, tol=tol)
        self.datasets = datasets

        self.dgl_collate = dgl.dataloading.GraphCollator().collate

    def train_dataloader(self):
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(split="val")

    def test_dataloader(self):
        return self._loader(split="test")

    def _loader(self, split, shuffle=False, drop_last=False):
        if self.distributed:
            sampler = DistributedSampler(
                seed=self.seed,
                dataset=self.datasets[split],
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = None
        else:
            sampler = None

        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=drop_last,
            worker_init_fn=pl_worker_init_function,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(self, items):
        G = self.dgl_collate(items)
        return chem.Molecule.from_dgl(G)
