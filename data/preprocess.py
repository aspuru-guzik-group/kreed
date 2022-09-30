import pathlib
import pickle

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

from kraitchman import rotate_to_principal_axes

SEED = 12049


def geom_unpacker():
    raw_dir = pathlib.Path(__file__).parent / "raw"
    raw_paths = list(raw_dir.glob("*.pkl"))
    for path in tqdm.tqdm(raw_paths, desc="Processing GEOM"):
        with open(path, "rb") as f:
            batch = pickle.load(f)
        yield batch


def preprocess_conformers(conformers):
    processed_data = []

    for conformer_idx, conformer_data in enumerate(conformers):
        xyz = np.array(conformer_data["xyz"])
        atom_nums, coords = xyz[:, 0], xyz[:, 1:]
        atom_nums = atom_nums.astype(np.int32)

        coords = rotate_to_principal_axes(atom_nums, coords)

        processed_data.append({
            "xyz": torch.tensor(coords),
            "atom_nums": torch.tensor(atom_nums),
            "geom_id": conformer_data["geom_id"],
        })

    return processed_data


def preprocess_geom():
    save_dir = pathlib.Path(__file__).parent / "processed"
    save_dir.mkdir(exist_ok=True)

    for batch_idx, batch in enumerate(geom_unpacker()):

        batch = list(batch.items())

        # [0.8 0.1 0.1] split
        train_subbatch, batch = train_test_split(batch, train_size=0.8, random_state=SEED)
        val_subbatch, test_subbatch = train_test_split(batch, train_size=0.5, random_state=(SEED + 1))

        subbatches = {
            "train": train_subbatch,
            "val": val_subbatch,
            "test": test_subbatch,
        }

        for split, subbatch in subbatches.items():
            subbatch = dict(subbatch)  # list of tuples -> dict

            processed_subbatch = []
            for smiles in subbatch:
                conformers = subbatch[smiles]["conformers"]
                processed_subbatch.extend(preprocess_conformers(conformers))
            torch.save(processed_subbatch, save_dir / f"{split}_{batch_idx}.pt")


if __name__ == "__main__":
    preprocess_geom()
