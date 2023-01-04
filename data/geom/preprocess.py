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
    geom_conformations = []
    for batch_idx, batch in enumerate(geom_unpacker()):

        for smiles, batch_metdata in batch.items():

            conformers = list(batch_metdata["conformers"])
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
                    "smiles_id": len(geom_smiles),
                })

            geom_smiles.append(smiles)
            geom_conformations.append(conformer_data)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(geom_smiles))
    torch.save(geom_conformations, save_dir / "conformations.pt")

    n_conformations = sum(len(x) for x in geom_conformations)
    print(f"Caching {n_conformations} conformations from {len(geom_smiles)} molecules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_conformers", type=int, default=30)
    args = parser.parse_args()

    preprocess_geom(args.n_conformers)
