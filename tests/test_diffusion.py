import torch

from src.datamodules import QM9Datamodule
from src.diffusion.dynamics import EquivariantDynamics
from src.diffusion.ddpm import EnEquivariantDDPM, EquivariantDDPMConfig
from src import utils


def test_diffusion():
    B = 64
    data = QM9Datamodule(100, B)
    G = data.datasets['train'][0]

    ddpm = EnEquivariantDDPM(
        EquivariantDDPMConfig(
            equivariance="reflect",
            timesteps=10,
        )
    )

    # Try loss computation
    loss = ddpm.simple_losses(G)

    ddpm.eval()
    loss = ddpm.nlls(G)

    # Try sampling
    G_init = G.local_var()
    G_init.ndata["xyz"] = torch.zeros_like(G_init.ndata["xyz"])  # safety
    G_gen, frames = ddpm.sample_p_G(G_init=G_init, keep_frames=[3, 4])
    assert len(frames) == 4

    utils.assert_zeroed_weighted_com(G_gen, G_gen.ndata["xyz"])
