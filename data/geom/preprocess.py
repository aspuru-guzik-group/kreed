import argparse
import pathlib

import msgpack
import torch
import tqdm


def geom_unpacker():
    path = pathlib.Path(__file__).parent / "raw" / "drugs_crude.msgpack"
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f)
        for batch in tqdm.tqdm(iter(unpacker), desc="Processing GEOM"):
            yield batch


def preprocess_geom(n_conformers):
    save_dir = pathlib.Path(__file__).parent / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    geom_smiles = []
    num_conformations = 0

    for batch_idx, batch in enumerate(geom_unpacker()):

        for smiles, batch_metadata in batch.items():
            smiles_id = len(geom_smiles)

            conformers = list(batch_metadata["conformers"])
            conformers.sort(key=lambda d: d["totalenergy"])
            conformers = conformers[:n_conformers]

            conformer_data = []

            for c in conformers:
                xyz = torch.tensor(c["xyz"])
                atom_nums, coords = xyz[:, 0].long(), xyz[:, 1:].float()

                conformer_data.append({
                    "xyz": coords,
                    "atom_nums": atom_nums,
                    "geom_id": c["geom_id"],
                    "smiles_id": smiles_id,
                })

            torch.save(conformer_data, save_dir / f"conformations_{smiles_id:06d}.pt")

            geom_smiles.append(smiles)
            num_conformations += len(conformer_data)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(geom_smiles))

    print(f"Caching {num_conformations} conformations from {len(geom_smiles)} molecules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_conformers", type=int, default=30)
    args = parser.parse_args()

    preprocess_geom(args.n_conformers)
