import torch

from src.datamodule import ConformerDatamodule
from src.diffusion.dynamics import EquivariantDynamics
from src.diffusion.ddpm import EquivariantDDPM, EquivariantDDPMConfig
from src import utils


def test_diffusion():
    data = ConformerDatamodule(
        dataset="qm9",
        seed=100,
        batch_size=512,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=8,
        distributed=False,
        tol=-1.0,
        p_drop_labels=0.0,
    )
    G = data.datasets['train'][0]

    ddpm = EquivariantDDPM(
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

    utils.assert_orthogonal_projection(G_gen, G_gen.ndata["xyz"])
