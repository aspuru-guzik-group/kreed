import datetime
import pathlib
from typing import List, Optional

import pydantic_cli
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import ConformerDatamodule
from src.diffusion import EquivariantDDPMConfig, LitEquivariantDDPM


class TrainEquivariantDDPMConfig(EquivariantDDPMConfig):
    """Configuration object for training the DDPM."""

    seed: int = 100
    debug: bool = False

    accelerator: str = "gpu"
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = "auto"

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

    max_epochs: int = 1500
    lr: float = 4e-4
    wd: float = 0.0
    puncond: float = 0.0
    pdropout_cond: List[float] = (0.0, 0.5)

    ema_decay: float = 0.999
    clip_grad_norm: bool = True

    # ================
    # Sampling Fields
    # ================

    check_samples_every_n_epochs: int = 1
    samples_visualize_n_mols: int = 3
    samples_assess_n_batches: int = 1
    samples_render_every_n_frames: int = 5

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_project: str = "train_edm"
    wandb_entity: Optional[str] = "matter-lab"
    wandb_run_id: str = None
    wandb_run_name: str = None

    checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_min: int = 10  # minutes

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train_ddpm(config: TrainEquivariantDDPMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Make directories
    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    if cfg.wandb_run_id is not None:
        ckpt_dir = ckpt_dir / cfg.wandb_run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        wd=cfg.wd,
        clip_grad_norm=cfg.clip_grad_norm,
        ema_decay=cfg.ema_decay,
        puncond=cfg.puncond,
        pdropout_cond=cfg.pdropout_cond,
        check_samples_every_n_epochs=cfg.check_samples_every_n_epochs,
        samples_visualize_n_mols=cfg.samples_visualize_n_mols,
        samples_assess_n_batches=cfg.samples_assess_n_batches,
        samples_render_every_n_frames=cfg.samples_render_every_n_frames,
        distributed=(cfg.strategy == "ddp"),
    )

    callbacks = [ModelSummary(max_depth=2)]

    if cfg.wandb:
        project = cfg.wandb_project + ("_debug" if cfg.debug else "")
        logger = WandbLogger(
            name=cfg.wandb_run_name,
            project=project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=ckpt_dir,
            config=dict(cfg),
            id=cfg.wandb_run_id,
            resume="allow",
        )
        callbacks.append(LearningRateMonitor())
    else:
        logger = False

    if cfg.checkpoint:
        ckpt_callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="epoch={epoch}-rmse={val/coord_rmse:.5f}",
                auto_insert_metric_name=False,
                monitor="val/coord_rmse",
                save_top_k=3,
                verbose=True,
                every_n_epochs=cfg.check_samples_every_n_epochs,
                save_on_train_epoch_end=False,
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                save_top_k=0,
                save_last=True,
                verbose=True,
                train_time_interval=datetime.timedelta(minutes=cfg.checkpoint_every_n_min),
            ),
        ]
        callbacks.extend(ckpt_callbacks)

    if cfg.debug:
        debug_kwargs = {
            "limit_train_batches": 10,
            "limit_val_batches": 10,
        }
    else:
        debug_kwargs = {}

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
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

    ckpt_path = ckpt_dir / "last.ckpt"
    ckpt_path = str(ckpt_path) if ckpt_path.exists() else None
    trainer.fit(model=ddpm, datamodule=data, ckpt_path=ckpt_path)

    if cfg.wandb and (ddpm.global_rank == 0):
        for callback in ckpt_callbacks:
            logger._scan_and_log_checkpoints(callback)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainEquivariantDDPMConfig, train_ddpm)
