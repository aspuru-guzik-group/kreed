from src import utils
from src.datamodule import ConformerDatamodule


def test_datamodule():
    datamodule = ConformerDatamodule(
        dataset="qm9",
        seed=100,
        batch_size=50,
        split_ratio=(0.8, 0.1, 0.1),
        num_workers=8,
        distributed=False,
        tol=-1.0,
    )

    batch = next(iter(datamodule.train_dataloader()))
    utils.assert_zeroed_com(batch, batch.coords)

    for M in batch.unbatch():
        assert abs(M.masses_normalized.sum().item() - 1.0) < 1e-5
