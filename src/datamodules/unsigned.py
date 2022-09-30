import pathlib

import dgl
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


class UnsignedCoordinateDataset(Dataset):

    def __init__(self, conformers, tol=-1.0):
        super().__init__()

        self.conformers = conformers
        self.tol = tol

    def __len__(self):
        return len(self.conformers)

    def __getitem__(self, idx):
        conformer = self.conformers[idx]
        xyz = conformer["xyz"]
        atom_nums = conformer["atom_nums"]

        unsigned_coords = torch.abs(xyz)
        labels = (xyz >= 0.0).float()

        mask = torch.logical_or(
            (atom_nums != 6),  # not carbon
            torch.any(unsigned_coords < self.tol, dim=-1)  # coordinate too close to axis
        )

        unsigned_coords[mask, :] = 0.0
        labels[mask, :] = 0.0

        G = dgl.rand_graph(atom_nums.shape[0], 0)
        G.ndata["atom_nums"] = conformer["atom_nums"]
        G.ndata["coords"] = unsigned_coords
        G.ndata["labels"] = labels
        G.ndata["mask"] = mask
        return G


class UnsignedCoordinateDatamodule(pl.LightningModule):

    def __init__(self, batch_size, num_workers=0, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data" / "processed"

        datasets = {"train": None, "val": None, "test": None}
        for split in datasets:
            entries = []
            for path in data_dir.glob(f"{split}_*.pt"):
                entries.extend(torch.load(path))
            datasets[split] = UnsignedCoordinateDataset(entries, **kwargs)
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
