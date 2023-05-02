import json
import pathlib

import torch
import torch.nn.functional as F

from src.diffusion.ddpm import EquivariantDDPM, EquivariantDDPMConfig
from src.experimental.train import LitEquivariantDDPM, TrainEquivariantDDPMConfig

assert TrainEquivariantDDPMConfig, "Used for torch.load-ing"


def clean_checkpoint(checkpoint_path, save_dir):
    checkpoint_path = pathlib.Path(checkpoint_path)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TODO: hacks for backward compatibility, remove later
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    if "grad_norm_queue" not in state_dict:
        print("Adding dummy grad_norm_queue buffer.")
        state_dict["grad_norm_queue"] = torch.zeros([50])
    if state_dict["edm.dynamics.embed_atom.weight"].shape[0] == 82:
        print("Padding atom embedding.")
        k = "edm.dynamics.embed_atom.weight"
        ema_k = "ema.ema_model.dynamics.embed_atom.weight"
        state_dict[k] = F.pad(state_dict[k], (0, 0, 0, 8))
        state_dict[ema_k] = F.pad(state_dict[ema_k], (0, 0, 0, 8))
    torch.save(checkpoint, "tmp/tmp.ckpt")
    checkpoint_path = "tmp/tmp.ckpt"
    # =========

    lit = LitEquivariantDDPM.load_from_checkpoint(checkpoint_path, map_location="cpu")

    # Sanity check
    assert all(
        not torch.allclose(p, p_ema)
        for p, p_ema in zip(lit.edm.parameters(), lit.ema.ema_model.parameters())
    )

    with open(save_dir / "config.json", "w+") as f:
        json.dump(lit.config.dict(), f, indent=2)
    torch.save(lit.state_dict(), save_dir / "edm.pt")  # Normal weights
    torch.save(lit.ema.ema_model.state_dict(), save_dir / "edm-ema.pt")  # EMA weights


if __name__ == "__main__":
    # TODO: expose through argparse
    clean_checkpoint("../../final_checkpoints/qm9.ckpt", "tmp/")

    # TODO: remove example
    with open("tmp/config.json", "r") as f:
        config = json.load(f)
    config = EquivariantDDPMConfig(**config)
    ddpm = EquivariantDDPM(config)
    print(ddpm.dynamics.embed_atom.weight)
    ddpm.load_state_dict(torch.load("tmp/edm-ema.pt"))
    print(ddpm.dynamics.embed_atom.weight)  # sanity check
