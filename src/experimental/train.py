import datetime
import pathlib
from typing import List, Optional

import pydantic_cli
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import ConformerDatamodule
from src.diffusion import EquivariantDDPMConfig, LitEquivariantDDPM
from src.experimental.ema import EMA

import torch
torch.set_float32_matmul_precision('medium')
class TrainEquivariantDDPMConfig(EquivariantDDPMConfig):
    """Configuration object for training the DDPM."""

    seed: int = 100
    debug: bool = False

    accelerator: str = "gpu"
    devices: int = 1
    strategy: Optional[str] = 'auto'

    # =================
    # Datamodule Fields
    # =================

    dataset: str = "qm9"

    batch_size: int = 512
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    num_workers: int = 8
    tol: float = -1.0

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = 3000
    lr: float = 2e-4
    puncond: float = 0.0

    ema_decay: float = 0.9999
    clip_grad_norm: bool = True

    # ================
    # Sampling Fields
    # ================

    check_samples_every_n_epoch: int = 1
    samples_visualize_n_mols: int = 3
    samples_assess_n_batches: int = 1
    samples_render_every_n_frames: int = 5
    n_eval_samples: int = 100

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_run_id: str = "tmp"

    checkpoint: bool = True
    checkpoint_dir: str = "."
    checkpoint_train_time_interval: int = 10  # minutes

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
    data = ConformerDatamodule(
        dataset=cfg.dataset,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        split_ratio=cfg.split_ratio,
        num_workers=cfg.num_workers,
        distributed=(cfg.strategy == "ddp"),
        tol=cfg.tol,
    )

    # Initialize and load model
    ddpm = LitEquivariantDDPM(
        config=cfg,
        lr=cfg.lr,
        puncond=cfg.puncond,
        clip_grad_norm=cfg.clip_grad_norm,
        check_samples_every_n_epoch=cfg.check_samples_every_n_epoch,
        samples_visualize_n_mols=cfg.samples_visualize_n_mols,
        samples_assess_n_batches=cfg.samples_assess_n_batches,
        samples_render_every_n_frames=cfg.samples_render_every_n_frames,
        n_eval_samples=cfg.n_eval_samples,
        distributed=(cfg.strategy == "ddp"),
    )

    if cfg.wandb:
        project = "train_edm" + ("_debug" if cfg.debug else "")
        logger = WandbLogger(
            project=project,
            entity=cfg.wandb_entity,
            log_model=True,
            save_dir=cfg.checkpoint_dir,
            config=dict(cfg),
            id=cfg.wandb_run_id,
            resume="allow",
        )
    else:
        logger = False

    callbacks = [
        ModelSummary(max_depth=2),
        EMA(decay=cfg.ema_decay, cpu_offload=True),
    ]

    if cfg.checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                monitor="val/nll",
                save_top_k=3,
                save_last=True,
                verbose=True,
                train_time_interval=datetime.timedelta(minutes=cfg.checkpoint_train_time_interval),
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

    ckpt_path = pathlib.Path(cfg.checkpoint_dir) / "last.ckpt"
    ckpt_path = str(ckpt_path) if ckpt_path.exists() else None
    trainer.fit(model=ddpm, datamodule=data, ckpt_path=ckpt_path)

    if cfg.wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainEquivariantDDPMConfig, train_ddpm)
