import pathlib

import dgl
import lightning_lite
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src import utils
from src.kraitchman import rotated_to_principal_axes


# ================================================================================================ #
#                                              Caches                                              #
# ================================================================================================ #

# QM9 constants

def _build_atom_map_cache():
    qm9_atoms = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)

    ztoi = torch.full([10], -100, dtype=torch.long)
    for i, z in enumerate(qm9_atoms):
        ztoi[z] = i
    return ztoi


_QM9_ATOMS_ZTOI = _build_atom_map_cache()


# Cache to optimize connected graph creation

def _build_edge_cache(max_nodes):
    cache = []
    for i in range(max_nodes):
        for j in range(i):
            cache.append([i, j])
            cache.append([j, i])
    return torch.tensor(cache, dtype=torch.long)


_EDGE_CACHE = _build_edge_cache(max_nodes=30)


# ================================================================================================ #
#                                         Data Handling                                            #
# ================================================================================================ #


class QM9Dataset(Dataset):

    def __init__(self, conformations, tol, zero_com, carbon_only, remove_Hs):
        super().__init__()

        self.conformations = conformations
        self.tol = tol
        self.zero_com = zero_com
        self.carbon_only = carbon_only
        self.remove_Hs = remove_Hs

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        txyz, geom_id = self.conformations[idx]
        atom_nums = torch.tensor(txyz[:, 0], dtype=torch.long)
        xyz = torch.tensor(txyz[:, 1:], dtype=torch.float)

        # Create a complete graph
        n = atom_nums.shape[0]
        edges = _EDGE_CACHE[:(n * (n - 1)), :]
        u, v = edges[:, 0], edges[:, 1]

        G = dgl.graph((u, v), num_nodes=n)
        G.ndata["atom_nums"] = atom_nums
        G.ndata["atom_ids"] = _QM9_ATOMS_ZTOI[atom_nums]
        G.ndata["xyz"] = xyz

        # Canonicalize conformation
        G = rotated_to_principal_axes(G)
        del xyz  # for safety

        # Retrieve unsigned coordinates for carbons that are not too close to coordinate axis
        abs_xyz = torch.abs(G.ndata["xyz"])
        abs_mask = (atom_nums == 6) & torch.any(abs_xyz >= self.tol, dim=-1)

        # Zero out non-carbons (later, zero out imaginary unsigned coordinates)
        abs_xyz[~abs_mask, :] = 0.0

        G.ndata["cond_labels"] = abs_xyz  # (N 3)
        G.ndata["cond_mask"] = abs_mask  # (N)

        # Potentially filter atoms
        filter_mask = torch.full_like(atom_nums, True, dtype=torch.bool)
        if self.remove_Hs:
            filter_mask &= (atom_nums != 1)
        if self.carbon_only:
            filter_mask &= (atom_nums == 6)
        if not torch.all(filter_mask):
            G = dgl.node_subgraph(G, filter_mask, store_ids=False)
        del n  # for safety

        if self.zero_com:
            # Center molecule coordinates to 0 CoM subspace
            G.ndata["xyz"] = utils.zeroed_com(G, G.ndata["xyz"])

        # Record GEOM ID
        G.ndata["id"] = torch.full((G.number_of_nodes(),), geom_id)  # hack to store graph-level data

        return G


class QM9Datamodule(pl.LightningDataModule):

    def __init__(
        self,
        seed,
        batch_size=64,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        tol=1e-5,
        zero_com=True,
        carbon_only=False,
        remove_Hs=False,
    ):
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data" / "qm9" / "processed"

        metadata = np.load(data_dir / "metadata.npy")
        start_indices = metadata[:, 0]

        coords = np.load(data_dir / "coords.npy")
        coords = np.split(coords, start_indices[1:])

        D = []
        for info, txyz in zip(metadata, coords):
            D.append((txyz, info[-1]))

        # Split by molecule
        splits = {"train": None, "val": None, "test": None}
        val_test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        splits["train"], D = train_test_split(D, train_size=split_ratio[0], random_state=seed)
        splits["val"], splits["test"] = train_test_split(D, train_size=val_test_ratio, random_state=(seed + 1))

        datasets = {}
        for split, conformations in splits.items():
            datasets[split] = QM9Dataset(
                conformations=conformations,
                tol=tol,
                zero_com=zero_com,
                carbon_only=carbon_only,
                remove_Hs=remove_Hs,
            )
        self.datasets = datasets

    @property
    def d_atom_vocab(self):
        return len(_QM9_ATOMS_ZTOI)

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
            worker_init_fn=lightning_lite.utilities.seed.pl_worker_init_function,
        )
