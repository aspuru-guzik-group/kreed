import pytest
import torch

from src.datamodule import ConformerDatamodule
from src.kraitchman import rotated_to_principal_axes


def test_kraitchman():
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
    G_0 = next(iter(data.train_dataloader()))
    G_0.ndata["xyz"].requires_grad = True

    G_rotated, moments = rotated_to_principal_axes(G_0, return_moments=True)

    assert moments.shape == (G_0.batch_size, 3)
    assert torch.all(moments[:, 0] >= moments[:, 1]) and torch.all(moments[:, 1] >= moments[:, 2])

    loss = G_rotated.ndata["xyz"].sum()
    loss.backward()

    grad = G_0.ndata["xyz"].grad
    assert (grad is not None) and torch.all(torch.isfinite(grad))
