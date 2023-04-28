import json
import pathlib

import torch

from src.experimental.train import LitEquivariantDDPM, TrainEquivariantDDPMConfig

assert TrainEquivariantDDPMConfig, "Used for torch.load-ing"


def clean_checkpoint(checkpoint_path, save_dir):
    checkpoint_path = pathlib.Path(checkpoint_path)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lit = LitEquivariantDDPM.load_from_checkpoint(checkpoint_path, map_location="cpu")

    # Sanity check
    assert all(
        not torch.allclose(p, p_ema)
        for p, p_ema in zip(lit.edm.parameters(), lit.ema.ema_model.parameters())
    )

    with open(save_dir / "config.json", "w+") as f:
        json.dump(lit.config.dict(), f, indent=2)
    torch.save(lit.edm.state_dict(), save_dir / "edm.pt")  # Normal weights
    torch.save(lit.ema.ema_model.state_dict(), save_dir / "edm-ema.pt")  # EMA weights


if __name__ == "__main__":
    # TODO: expose through argparse
    clean_checkpoint("checkpoints/last.ckpt", "tmp/")
