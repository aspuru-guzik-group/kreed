import pathlib

import msgpack
import numpy as np
import tqdm


def qm9_unpacker():
    path = pathlib.Path(__file__).parent / "raw" / "qm9_crude.msgpack"
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f)
        for batch in tqdm.tqdm(iter(unpacker), desc="Processing QM9"):
            yield batch


def preprocess_qm9():
    save_dir = pathlib.Path(__file__).parent / "processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    qm9_metadata = []
    qm9_smiles = []
    qm9_coords = []

    for batch_idx, batch in enumerate(qm9_unpacker()):

        for smiles, batch_metadata in batch.items():
            conformers = list(batch_metadata["conformers"])
            conformers.sort(key=lambda d: d["totalenergy"])
            c = conformers[0]

            info = [
                len(qm9_coords),  # start index
                len(qm9_smiles),  # SMILES id
                c["geom_id"],  # GEOM id
            ]

            coords = np.array(c["xyz"], dtype=np.float32)  # (n, 4) where txyz[i] = [atom_type, x, y, z]

            qm9_metadata.append(info)
            qm9_smiles.append(smiles)
            qm9_coords.append(coords)

    # ==============
    # Cache to files
    # ==============

    qm9_metadata = np.array(qm9_metadata, dtype=np.int64)
    np.save(save_dir / "metadata.npy", qm9_metadata)

    with open(save_dir / "smiles.txt", "w+") as f:
        f.write("\n".join(qm9_smiles))

    num_coords = len(qm9_coords)
    np.save(save_dir / "coords.npy", np.vstack(qm9_coords))

    # Done!
    print(f"Cached {num_coords} conformations from {len(qm9_smiles)} molecules.")


if __name__ == "__main__":
    preprocess_qm9()
