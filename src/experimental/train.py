import pathlib

import pydantic_cli
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import QM9Datamodule
from src.diffusion.configs import TrainEquivariantDDPMConfig
from src.diffusion.lit import LitEquivariantDDPM


def train_ddpm(config: TrainEquivariantDDPMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    geom = QM9Datamodule(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        split_ratio=cfg.split_ratio,
        num_workers=cfg.num_workers,
        tol=cfg.tol,
        zero_com=(cfg.equivariance == "e3"),
        carbon_only=cfg.carbon_only,
        remove_Hs=cfg.remove_Hs,
    )

    # Initialize and load model
    ddpm = LitEquivariantDDPM(config=cfg)

    if cfg.wandb:
        project = "train_edm" + ("_debug" if cfg.debug else "")
        logger = WandbLogger(project=project, log_model=True, save_dir=str(log_dir))
        logger.experiment.config.update(dict(cfg))
    else:
        logger = False

    callbacks = [ModelSummary(max_depth=2)]

    if cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=wandb.run.dir,
                monitor="val/nll",
                save_top_k=3,
                save_last=True,
            )
        )

    if cfg.debug:
        debug_kwargs = {
            "overfit_batches": 1000,
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

    trainer.fit(model=ddpm, datamodule=geom)
    trainer.validate(model=ddpm, datamodule=geom)
    trainer.test(model=ddpm, datamodule=geom)

    if cfg.wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainEquivariantDDPMConfig, train_ddpm)
