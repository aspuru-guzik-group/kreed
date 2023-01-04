import pathlib

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import sys
sys.path.append('.')
from src.datamodules import GEOMDatamodule
from src.experiments.pl_ddpm import PlEnEquivariantDiffusionModel
from pytorch_lightning.profilers import AdvancedProfiler


def main():
    # Downscaled training script
    pl.seed_everything(100)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    geom = GEOMDatamodule(
        seed=100,
        batch_size=64,
        num_workers=12,
        split='100k',
    )

    # Initialize and load model
    model = PlEnEquivariantDiffusionModel(
        d_egnn_hidden=256,
        n_egnn_layers=4,
        timesteps=40,
        n_sample_metric_batches=1,
        clip_grad_norm=False,
        n_visualize_samples=1,
    )

    logger = WandbLogger(project="train_geom_debug", log_model=False, save_dir=str(log_dir))

    checkpointer = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val/nll",
        save_top_k=1,
        save_last=True,
    )

    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs2")
    trainer = pl.Trainer(
        callbacks=[checkpointer],
        logger=logger,
        log_every_n_steps=100,
        min_epochs=1,
        max_epochs=1000,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        limit_train_batches=1.0,
        # limit_val_batches=0,
        # limit_test_batches=0,
        deterministic=False,
        # profiler='simple',
        # profiler=profiler,
        # enable_checkpointing=False,
        # overfit_batches=1,
    )

    trainer.fit(model=model, datamodule=geom)
    
    # print('visualizing...')
    # example = geom.datasets['train'][0]
    # model._visualize_and_check_samples(example, 'test', 1)

    trainer.test(model=model, datamodule=geom)

    wandb.finish()


if __name__ == "__main__":
    main()
