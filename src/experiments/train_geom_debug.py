import pathlib

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import GEOMDatamodule
from src.experiments.pl_ddpm import PlEnEquivariantDiffusionModel


def main():
    # Downscaled training script
    pl.seed_everything(100)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    datamodule = GEOMDatamodule(
        seed=100,
        batch_size=10,
        num_workers=4,
    )

    # Initialize and load model
    model = PlEnEquivariantDiffusionModel(
        d_egnn_hidden=32,
        n_egnn_layers=2,
        timesteps=20,
        n_sample_batches=1,
    )

    # logger = WandbLogger(project="train_geom_debug", log_model=False, save_dir=str(log_dir))

    checkpointer = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_nll",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        callbacks=[checkpointer],
        logger=False,
        min_epochs=100,
        max_epochs=100,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        limit_train_batches=5,
        limit_val_batches=5,
        limit_test_batches=5,
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
