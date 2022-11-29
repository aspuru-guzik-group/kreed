import torch

from src.datamodules import GEOMDatamodule
from src.diffusion import EGNNDynamics, EnEquivariantDDPM
from src.diffusion.distributions import assert_centered_mean


def test_diffusion():
    geom = GEOMDatamodule(seed=0, batch_size=64)
    G_0 = next(iter(geom.train_dataloader()))

    egnn = EGNNDynamics(
        d_atom_vocab=geom.d_atom_vocab,
        d_hidden=256,
        n_layers=4,
    )

    ddpm = EnEquivariantDDPM(
        dynamics=egnn,
        timesteps=10,
        noise_shape="polynomial_2",
        noise_precision=0.008,
        loss_type="L2",
    )

    # Try loss computation
    loss = ddpm(G_0)
    print(loss)

    ddpm.eval()
    loss = ddpm(G_0)
    print(loss)

    # Try sampling
    G_init = G_0.local_var()
    G_init.ndata["xyz"] = torch.zeros_like(G_init.ndata["xyz"])  # safety
    G_gen, frames = ddpm.sample_p_G0(G_init=G_init, keep_frames=[200])
    print(G_gen)

    assert_centered_mean(G_gen, G_gen.ndata["xyz"])


if __name__ == "__main__":
    test_diffusion()
