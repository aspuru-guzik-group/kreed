import argparse
import pathlib
import pickle

import torch
import tqdm


def geom_unpacker():
    raw_dir = pathlib.Path(__file__).parent / "raw"
    raw_paths = list(raw_dir.glob("*.pkl"))
    for path in tqdm.tqdm(raw_paths, desc="Processing GEOM"):
        with open(path, "rb") as f:
            batch = pickle.load(f)
        yield batch


def preprocess_geom(n_conformers, remove_Hs):
    save_dir = pathlib.Path(__file__).parent / "processed" / f"confs_{n_conformers}{('_no_Hs' if remove_Hs else '')}"
    save_dir.mkdir(parents=True, exist_ok=True)

    geom_smiles = []
    geom_conformations = []
    geom_atom_nums = set()

    for batch_idx, batch in enumerate(geom_unpacker()):

        for smiles, batch_metdata in batch.items():

            conformers = list(batch_metdata["conformers"])
            conformers.sort(key=lambda d: d["totalenergy"])
            conformers = conformers[:n_conformers]

            for c in conformers:
                xyz = torch.tensor(c["xyz"])
                atom_nums, coords = xyz[:, 0].int(), xyz[:, 1:].float()

                if remove_Hs:
                    mask = (atom_nums != 1)
                    atom_nums = atom_nums[mask]
                    coords = coords[mask]

                geom_atom_nums.update(atom_nums.tolist())

                geom_conformations.append({
                    "xyz": coords,
                    "atom_nums": atom_nums,
                    "geom_id": c["geom_id"],
                    "smiles_id": len(geom_smiles),
                })

            geom_smiles.append(smiles)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(geom_smiles))
    with open(save_dir / "atoms.txt", "w+") as f:
        geom_atom_nums = list(map(str, sorted(geom_atom_nums)))
        f.write("\n".join(geom_atom_nums))
    torch.save(geom_conformations, save_dir / "conformations.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_conformers", type=int, default=30)
    parser.add_argument("--remove_Hs", action="store_true")
    args = parser.parse_args()

    preprocess_geom(**vars(args))
