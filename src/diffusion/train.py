import pathlib

import pydantic_cli
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append('.')

from src.datamodules import GEOMDatamodule
from src.diffusion.configs import TrainEquivariantDDPMConfig
from src.diffusion.model import LitEquivariantDDPM

def train_ddpm(config: TrainEquivariantDDPMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    overfit_samples = 1000 if cfg.debug else None

    # Load data
    geom = GEOMDatamodule(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        split_ratio=cfg.split_ratio,
        num_workers=cfg.num_workers,
        tol=cfg.tol,
        center_mean=(cfg.equivariance=='rotation'),
        overfit_samples=overfit_samples,
        carbon_only=cfg.carbon_only,
    )

    print("Datamodule loaded.")

    # Initialize and load model
    ddpm = LitEquivariantDDPM(
        config=cfg,
        )
    print("Model loaded.")

    if cfg.wandb:
        project = "train_equiv_ddpm" + ("_debug" if cfg.debug else "")
        logger = WandbLogger(project=project, log_model=True, save_dir=str(log_dir))
        logger.experiment.config.update(dict(cfg))
    else:
        logger = False

    callbacks = [ModelSummary(max_depth=2)]

    if cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=wandb.run.dir,
                monitor="val/loss",
                save_top_k=3,
                save_last=True,
            )
        )

    if cfg.debug:
        debug_kwargs = {
            "limit_train_batches": 10000,
            "limit_val_batches": 0,
            "limit_test_batches": 0,
        }
    else:
        debug_kwargs = {}
    
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        callbacks=callbacks,
        enable_checkpointing=cfg.checkpoint,
        logger=logger,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.progress_bar,
        num_sanity_val_steps=1,
        **debug_kwargs,
    )

    print("Trainer loaded.")
    print("Beginning training...")

    trainer.fit(model=ddpm, datamodule=geom)
    trainer.validate(model=ddpm, datamodule=geom)
    trainer.test(model=ddpm, datamodule=geom)

    if cfg.wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainEquivariantDDPMConfig, train_ddpm)
