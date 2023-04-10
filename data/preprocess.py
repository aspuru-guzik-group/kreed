import argparse
import pathlib
import pickle

import msgpack
import numpy as np
import tqdm


def unpacker(fpath):
    with open(fpath, "rb") as f:
        if fpath.suffix == ".msgpack":
            unpacker = msgpack.Unpacker(f)
            for batch in tqdm.tqdm(iter(unpacker), desc="Processing GEOM"):
                yield batch
        else:
            yield pickle.load(f)


def preprocess_dataset(dataset, num_conformers, min_atoms, debug):
    root = pathlib.Path(__file__).parent / dataset

    dataset_metadata = []
    dataset_smiles = []
    dataset_coords = []
    start_idx = 0

    if dataset == "qm9":
        fpath = root / "raw" / "qm9_crude.msgpack"
    elif dataset == "geom":
        fpath = root / "raw" / ("drugs_1k.pkl" if debug else "drugs_crude.msgpack")
    else:
        raise NotImplementedError()

    for batch_idx, batch in enumerate(unpacker(fpath)):

        for smiles, batch_metadata in batch.items():
            conformers = list(batch_metadata["conformers"])
            conformers.sort(key=lambda d: d["totalenergy"])
            conformers = conformers[:num_conformers]

            for c in conformers:
                info = [
                    start_idx,  # start index
                    len(smiles),  # SMILES id
                    c["geom_id"],  # GEOM id
                ]

                # (N 4) where coords[i] = [atom_type, x, y, z]
                txyz = np.array(c["xyz"], dtype=np.float32)

                # Remove molecules that are too small (e.g., for QM9)
                if txyz.shape[0] < min_atoms:
                    pass
                else:
                    start_idx += txyz.shape[0]
                    dataset_metadata.append(info)
                    dataset_coords.append(txyz)
            dataset_smiles.append(smiles)

    # Cache to files
    save_dir = root / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(dataset_smiles))

    np.save(str(save_dir / "metadata.npy"), np.array(dataset_metadata, dtype=np.int64))
    np.save(str(save_dir / "coords.npy"), np.concatenate(dataset_coords, axis=0))

    # Done!
    print(f"Cached {len(dataset_coords)} conformations from {len(dataset_smiles)} molecules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["geom", "qm9"])
    parser.add_argument("--num_conformers", type=int)
    parser.add_argument("--min_atoms", type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    preprocess_dataset(**vars(args))
