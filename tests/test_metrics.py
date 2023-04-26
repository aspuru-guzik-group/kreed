import pytest
import torch
from scipy.stats import ortho_group

from src import kraitchman
from src.datamodule import ConformerDatamodule
from src.metrics import evaluate_prediction


def randomly_move_rigidly(M):
    b = torch.randn([3], dtype=torch.float)
    A = torch.tensor(ortho_group.rvs(3), dtype=torch.float)
    coords = M.coords @ A.T + b
    return M.replace(coords=coords)


@pytest.fixture
def conformer_datamodule():
    return ConformerDatamodule(
        dataset="qm9",
        seed=100,
        batch_size=50,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=8,
        distributed=False,
        tol=-1.0,
    )


def test_kraitchman(conformer_datamodule):
    batch = next(iter(conformer_datamodule.train_dataloader()))
    for M in batch.unbatch():
        M1 = randomly_move_rigidly(M)
        M2 = randomly_move_rigidly(M)
        coords1, moments1 = kraitchman.rotated_to_principal_axes(M1.coords, M1.masses, return_moments=True)
        coords2, moments2 = kraitchman.rotated_to_principal_axes(M2.coords, M2.masses, return_moments=True)

        assert torch.allclose(coords1.abs(), coords2.abs(), rtol=1e-5, atol=1e-4)
        assert torch.allclose(moments1, moments2, rtol=1e-5, atol=1e-4)


def test_evaluate_prediction(conformer_datamodule):
    batch = next(iter(conformer_datamodule.train_dataloader()))
    for M in batch.unbatch():
        M_pred = randomly_move_rigidly(M)
        metrics = evaluate_prediction(M_pred=M_pred, M_true=M)

        try:
            M.smiles()
        except:
            assert metrics["correctness"] == 1.0
        assert metrics["moments_rmse"] < 1e-3
        assert metrics["unsigned_coords_rmse"] < 1e-4
        assert metrics["coord_rmse"] < 1e-4
