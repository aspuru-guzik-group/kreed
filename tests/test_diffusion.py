import torch

from src.datamodules import GEOMDatamodule
from src.modules import EGNNDynamics, EnEquivariantDDPM, KraitchmanClassifier
from src.modules.distributions import assert_centered_mean


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
        classifier=None,
        timesteps=10,
        noise_shape="polynomial_2",
        noise_precision=0.008,
        loss_type="L2",
    )

    # Try loss computation
    loss = ddpm(G_0)
    assert loss.shape == (64,)

    ddpm.eval()
    loss = ddpm(G_0)
    assert loss.shape == (64,)

    # Try sampling
    G_init = G_0.local_var()
    G_init.ndata["xyz"] = torch.zeros_like(G_init.ndata["xyz"])  # safety
    G_gen, frames = ddpm.sample_p_G0(G_init=G_init, keep_frames=[3, 4])
    assert len(frames) == 4

    assert_centered_mean(G_gen, G_gen.ndata["xyz"])


def test_classifier():
    geom = GEOMDatamodule(seed=0, batch_size=64)
    G_0 = next(iter(geom.train_dataloader()))

    xyz = G_0.ndata["xyz"]
    xyz_pert = xyz + torch.randn_like(xyz)

    G_corrupt = G_0.local_var()
    G_corrupt.ndata["xyz"] = xyz_pert

    clf = KraitchmanClassifier(scale=1.0, stable=True)
    assert clf.grad_log_p_y_given_Gt(G_corrupt).shape == xyz.shape
