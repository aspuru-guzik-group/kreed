import pathlib

import dgl
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class GEOMDataset(Dataset):

    def __init__(self, conformations, remove_Hs, tol):
        super().__init__()

        self.conformations = conformations
        self.remove_Hs = remove_Hs
        self.tol = tol

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        conformer = self.conformations[idx]

        xyz = conformer["xyz"]
        atom_nums = conformer["atom_nums"]
        n = atom_nums.shape[0]

        if self.remove_Hs:
            nonH_mask = (atom_nums != 1)
            xyz = xyz[nonH_mask]
            atom_nums = [nonH_mask]

        abs_xyz = torch.abs(xyz)

        sign_mask = torch.logical_or(
            (atom_nums != 6),  # not carbon
            torch.any(abs_xyz < self.tol, dim=-1),  # coordinate too close to axis
        )

        abs_xyz[sign_mask, :] = 0.0

        # create a complete graph
        nodes = torch.arange(n)
        u, v = torch.meshgrid(nodes, nodes)
        u, v = u.flatten(), v.flatten()

        G = dgl.graph((u, v), num_nodes=n)
        G.ndata["atom_nums"] = atom_nums
        G.ndata["xyz"] = xyz
        G.ndata["abs_xyz"] = abs_xyz
        return G


class GEOMDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        seed,
        batch_size=64,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=0,
        remove_Hs=False,
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
            datasets[n] = GEOMDataset(conformations, remove_Hs=remove_Hs, tol=tol)
        self.datasets = datasets

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
