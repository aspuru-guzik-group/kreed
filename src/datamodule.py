import pathlib
import dgl
import einops
import lightning_lite
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src import chem, kraitchman

def _build_edge_cache(max_nodes):
    cache = []
    for i in range(max_nodes):
        for j in range(i):
            cache.append([i, j])
            cache.append([j, i])
    return torch.tensor(cache, dtype=torch.long)


_EDGE_CACHE = _build_edge_cache(max_nodes=200)

class ConformerDataset(Dataset):

    def __init__(self, conformations, tol, p_drop_labels):
        super().__init__()

        self.conformations = conformations
        self.tol = tol
        self.p_drop_labels = p_drop_labels

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        txyz, geom_id = self.conformations[idx]

        n = txyz.shape[0]
        xyz = torch.tensor(txyz[:, 1:], dtype=torch.float)  # (N 3)
        atom_nums = torch.tensor(txyz[:, :1], dtype=torch.long)  # (N 1)

        # Create a complete graph
        n = atom_nums.shape[0]
        edges = _EDGE_CACHE[:(n * (n - 1)), :]
        u, v = edges[:, 0], edges[:, 1]

        G = dgl.graph((u, v), num_nodes=n)
        G.ndata["atom_nums"] = atom_nums  # (N)
        G.ndata["atom_ids"] = chem.ATOM_ZTOI[atom_nums]  # (N)
        G.ndata["atom_masses"] = chem.ATOM_MASSES[atom_nums]  # (N)
        G.ndata["xyz"] = xyz  # (N 3)

        # Canonicalize conformation
        G, moments = kraitchman.rotated_to_principal_axes(G, return_moments=True)
        xyz = G.ndata["xyz"]

        # Retrieve unsigned coordinates for isotopically abundant atoms that
        # are not too close to coordinate axis
        cond_mask = torch.isin(atom_nums, chem.ISOTOPICALLY_ABUNDANT_ATOMS) & (xyz.abs() >= self.tol)
        if self.p_drop_labels > 0.0:
            # Drop labels with probability p_drop_labels
            # sample a mask of shape (N 3) with values in [0, 1]
            mask = torch.rand_like(xyz) > self.p_drop_labels
            cond_mask = cond_mask & mask
        
        cond_labels = torch.where(cond_mask, xyz.abs(), 0.0)
        G.ndata["cond_mask"] = cond_mask  # (N 3)
        G.ndata["cond_labels"] = cond_labels  # (N 3)

        # Record graph-level data (hack: store as node features)
        G.ndata["moments"] = einops.repeat(moments, "o d -> (n o) d", n=n)  # (N 3)
        G.ndata["id"] = torch.full_like(atom_nums, geom_id, dtype=torch.long)

        # TODO: commented out since we require all atoms to be present for
        #  atom-weighted CoM to make sense
        # Potentially filter atoms
        # filter_mask = torch.full_like(atom_nums, True, dtype=torch.bool)
        # if self.remove_Hs:
        #     filter_mask &= (atom_nums != 1)
        # if self.carbon_only:
        #     filter_mask &= (atom_nums == 6)
        # if not torch.all(filter_mask):
        #     G = dgl.node_subgraph(G, filter_mask, store_ids=False)

        return G


class ConformerDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        dataset,
        seed,
        batch_size,
        split_ratio,
        num_workers,
        distributed,
        tol,
        p_drop_labels,
    ):
        super().__init__()

        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.tol = tol
        self.p_drop_labels = p_drop_labels

        # Load data
        data_dir = pathlib.Path(__file__).parents[1] / "data" / dataset / "processed"
        metadata = np.load(str(data_dir / "metadata.npy"))
        coords = np.load(str(data_dir / "coords.npy"))

        # Unbind coordinates
        start_indices = metadata[:, 0]
        coords = np.split(coords, start_indices[1:])

        # Group conformations by molecule
        D = []
        for info, txyz in zip(metadata, coords):
            _, molecule_id, geom_id = list(info)
            if molecule_id >= len(D):
                D.append([])
            D[molecule_id].append((txyz, geom_id))

        # Create train/val/test split
        splits = {"train": None, "val": None, "test": None}
        val_test_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        splits["train"], D = train_test_split(D, train_size=split_ratio[0], random_state=seed)
        splits["val"], splits["test"] = train_test_split(D, train_size=val_test_ratio, random_state=seed+1)

        # Create PyTorch datasets
        datasets = {}
        for split, D_split in splits.items():
            conformations = []
            for mol_conformers in D_split:
                conformations.extend(mol_conformers)
            datasets[split] = ConformerDataset(conformations, tol=tol, p_drop_labels=p_drop_labels)
        self.datasets = datasets

    def train_dataloader(self):
        return self._loader(split="train", shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(split="val", shuffle=False)

    def test_dataloader(self):
        return self._loader(split="test", shuffle=False)

    def _loader(self, split, shuffle, drop_last=False):
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
            worker_init_fn=lightning_lite.utilities.seed.pl_worker_init_function,
            collate_fn=dgl.dataloading.GraphCollator().collate,
            pin_memory=True,
        )
