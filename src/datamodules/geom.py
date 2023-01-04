import pathlib
import random
import numpy as np
import dgl
import lightning_lite
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import src.modules.distributions as dists
from src.kraitchman import rotated_to_principal_axes


# ================================================================================================ #
#                                              Caches                                              #
# ================================================================================================ #

# GEOM constants

def _build_atom_map_cache():
    geom_atoms = torch.tensor([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83], dtype=torch.long)

    ztoi = torch.full([84], -100, dtype=torch.long)
    for i, z in enumerate(geom_atoms):
        ztoi[z] = i
    return ztoi


_GEOM_ATOMS_ZTOI = _build_atom_map_cache()


# Cache to optimize connected graph creation

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


class GEOMDataset(Dataset):

    def __init__(self, conformations, tol):
        super().__init__()

        self.conformations = conformations
        self.tol = tol

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        conformer = self.conformations[idx]

        geom_id = int(conformer[0][0])
        atom_nums = torch.tensor(conformer[:, 1], dtype=torch.long)
        xyz = torch.tensor(conformer[:, 2:], dtype=torch.float)

        n = atom_nums.shape[0]

        # Create a complete graph
        edges = _EDGE_CACHE[:(n * (n - 1)), :]
        u, v = edges[:, 0], edges[:, 1]

        G = dgl.graph((u, v), num_nodes=n)
        G.ndata["atom_nums"] = atom_nums
        G.ndata["atom_ids"] = _GEOM_ATOMS_ZTOI[atom_nums]
        G.ndata["xyz"] = xyz

        # Canonicalize conformation
        G = rotated_to_principal_axes(G)

        # Center molecule coordinates to 0 CoM subspace
        G.ndata["xyz"] = dists.centered_mean(G, G.ndata["xyz"])

        # Retrieve unsigned coordinates
        abs_xyz = torch.abs(G.ndata["xyz"])

        abs_mask = torch.logical_and(
            (atom_nums == 6),  # carbon
            torch.any(abs_xyz >= self.tol, dim=-1),  # coordinate not too close to axis
        )

        abs_xyz[~abs_mask, :] = 0.0

        G.ndata["abs_xyz"] = abs_xyz
        G.ndata["abs_mask"] = abs_mask

        G.ndata['signs'] = torch.where(abs_xyz == 0.0, 0.0, G.ndata['xyz'] / abs_xyz)

        G.ndata['free_xyz'] = torch.where(G.ndata['signs'] == 0.0, G.ndata['xyz'], 0.0)
        G.ndata['free_mask'] = ~abs_mask

        # Convert atom number to idx
        G.ndata["atom_nums"] = self.ztoi[G.ndata["atom_nums"]]

        # Record GEOM ID
        G.ndata["id"] = torch.full((n,), geom_id)  # hack to store graph-level data

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

        # This is a single tensor (*, 7) containing all data, where each row describes an atom
        # idx 0: smiles_id of the molecule it belongs to
        # idx 1: number of atoms in the molecule it belongs to
        # idx 2: geom_id of the conformer it belongs to
        # idx 3: atom type
        # idx 4-6: xyz
        with open(data_dir / "conformations.npy", 'rb') as f:
            conformations = np.load(f)
        # with open(data_dir / "number_atoms.npy", 'rb') as f:
        #     self.number_atoms = np.load(f)
        smiles_id = conformations[:, 0].astype(int)
        conformers = conformations[:, 1:]

        # Get ids corresponding to new molecules
        split_indices = np.nonzero(smiles_id[:-1] - smiles_id[1:])[0] + 1

        conformers_by_mol = np.split(conformers, split_indices)
        # Split by molecule
        splits = {"train": None, "val": None, "test": None}
        val_test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        splits["train"], conformers_by_mol = train_test_split(conformers_by_mol, train_size=split_ratio[0], random_state=seed)
        splits["val"], splits["test"] = train_test_split(conformers_by_mol, train_size=val_test_ratio, random_state=(seed + 1))

        datasets = {}
        for n, conformations in splits.items():

            all_conformations = []
            for mol in splits['train']:
                n = int(mol[0][0])
                mol_confs = mol[:, 1:]
                all_conformations.extend(np.split(mol_confs, len(mol) / n))

            datasets[n] = GEOMDataset(all_conformations, tol=tol)
        self.datasets = datasets

    @property
    def d_atom_vocab(self):
        return len(_GEOM_ATOMS_ZTOI)

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
