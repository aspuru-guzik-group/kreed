import pytest
import torch

from src.datamodules import QM9Datamodule
from src.kraitchman import rotated_to_principal_axes


@pytest.mark.parametrize("stable", [True, False])
def test_kraitchman(stable):
    geom = QM9Datamodule(seed=0, batch_size=64)
    G_0 = next(iter(geom.train_dataloader()))
    G_0.ndata["xyz"].requires_grad = True

    G_rotated, moments = rotated_to_principal_axes(G_0, stable=stable, return_moments=True)

    assert moments.shape == (G_0.batch_size, 3)
    assert torch.all(moments[:, 0] >= moments[:, 1]) and torch.all(moments[:, 1] >= moments[:, 2])

    loss = G_rotated.ndata["xyz"].sum()
    loss.backward()

    grad = G_0.ndata["xyz"].grad
    assert (grad is not None) and torch.all(torch.isfinite(grad))
