import argparse
import pathlib

import msgpack
import torch
import tqdm
import numpy as np


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
    geom_number_atoms = []
    for batch_idx, batch in enumerate(geom_unpacker()):

        for smiles, batch_metadata in batch.items():

            conformers = list(batch_metadata["conformers"])
            conformers.sort(key=lambda d: d["totalenergy"])
            conformers = conformers[:n_conformers]

            for c in conformers:

                # atom_type, x, y, z
                txyz = np.array(c['xyz']).astype(float) # (n, 4)

                n = len(txyz)

                size = n * np.ones((n, 1), dtype=float)
                smiles_id = len(geom_smiles) * np.ones((n, 1), dtype=float)
                geom_id = c['geom_id'] * np.ones((n, 1), dtype=float)

                example = np.hstack( (smiles_id, size, geom_id, txyz) ) # (n, 7)

                geom_conformations.append(example)
            
            geom_number_atoms.append(n)
            geom_smiles.append(smiles)
    
    num_conformations = len(geom_conformations)
    geom_conformations = np.vstack(geom_conformations)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(geom_smiles))
    
    with open(save_dir / 'conformations.npy', 'wb') as f:
        np.save(f, geom_conformations)

    print(f"Caching {num_conformations} conformations from {len(geom_smiles)} molecules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_conformers", type=int, default=30)
    args = parser.parse_args()

    preprocess_geom(args.n_conformers)
