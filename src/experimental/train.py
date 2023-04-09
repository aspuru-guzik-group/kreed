import pathlib
from typing import List, Optional

import pydantic_cli
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import QM9Datamodule, GEOMDatamodule
from src.diffusion import LitEquivariantDDPM, LitEquivariantDDPMConfig
from datetime import timedelta
import os

import torch
torch.set_float32_matmul_precision('medium')
class TrainEquivariantDDPMConfig(LitEquivariantDDPMConfig):
    """Configuration object for training the DDPM."""

    seed: int = 100
    debug: bool = False

    accelerator: str = "gpu"
    devices: int = 1
    strategy: Optional[str] = 'auto'

    # =================
    # Datamodule Fields
    # =================

    dataset: str = "geom"

    batch_size: int = 64
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    num_workers: int = 4
    tol: float = -1.0

    carbon_only: bool = False
    remove_Hs: bool = False

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False

    checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_train_time_interval: int = 10 # minutes

    run_id: str = "default"

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train_ddpm(config: TrainEquivariantDDPMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    if cfg.dataset == 'geom':
        data = GEOMDatamodule(
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            split_ratio=cfg.split_ratio,
            num_workers=cfg.num_workers,
            tol=cfg.tol,
            zero_com=(cfg.equivariance == "e3"),
            carbon_only=cfg.carbon_only,
            remove_Hs=cfg.remove_Hs,
        )
    elif cfg.dataset == 'qm9':
        data = QM9Datamodule(
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
        logger = WandbLogger(project=project,
                             log_model=True,
                             save_dir='.',
                             version=cfg.run_id,
                             config=dict(cfg),
                             resume="allow",

        )
    else:
        logger = False

    callbacks = [ModelSummary(max_depth=2)]

    if cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir + "/" + cfg.run_id,
                monitor="val/nll",
                save_top_k=3,
                save_last=True,
                verbose=True,
                train_time_interval=timedelta(minutes=cfg.checkpoint_train_time_interval),
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
        num_sanity_val_steps=0,
        **debug_kwargs,
    )

    checkpoint_path = cfg.checkpoint_dir + "/" + cfg.run_id + "/last.ckpt"
    if os.path.exists(checkpoint_path):
        trainer.fit(model=ddpm, datamodule=data, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model=ddpm, datamodule=data)

    if cfg.wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainEquivariantDDPMConfig, train_ddpm)
