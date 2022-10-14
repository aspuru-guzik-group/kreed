import pathlib

import dgl
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class GEOMDataset(Dataset):

    def __init__(self, conformers, atoi, tol=-1.0):
        super().__init__()

        self.conformers = conformers
        self.atoi = atoi
        self.tol = tol

        self.d_vocab = (self.atoi >= 0).int().sum().item()

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
            torch.any(unsigned_coords < self.tol, dim=-1),  # coordinate too close to axis
        )

        atom_ids, atom_counts = torch.unique(atom_nums, return_counts=True)
        formula = torch.zeros(self.d_vocab, dtype=torch.long)
        formula[self.atoi[atom_ids.long()]] = atom_counts
        formula = formula.int()

        atom_nums = atom_nums[mask]
        unsigned_coords = unsigned_coords[mask, :]
        labels = labels[mask, :]

        G = dgl.rand_graph(unsigned_coords.shape[0], 0)
        G.ndata["atom_nums"] = atom_nums
        G.ndata["coords"] = unsigned_coords
        G.ndata["labels"] = labels
        return G, formula


class GEOMDatamodule(pl.LightningDataModule):

    def __init__(
        self,
        seed,
        batch_size,
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

        with open(data_dir / "atoms.txt") as f:
            atom_vocab = [int(z.strip()) for z in f.readlines()]
            atom_vocab.sort()
            if remove_Hs:
                assert atom_vocab[0] == 1
                atom_vocab = atom_vocab[1:]  # remove H from atom vocab
        self.d_vocab = len(atom_vocab)


        conformations = torch.load(data_dir / "conformations.pt")




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
