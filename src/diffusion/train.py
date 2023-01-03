import pathlib

import pydantic_cli
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import GEOMDatamodule
from src.diffusion.configs import TrainEnEquivariantDDPMConfig
from src.diffusion.model import LitEnEquivariantDDPM


def train_ddpm(config: TrainEnEquivariantDDPMConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed)

    # Make directories
    root = pathlib.Path(__file__).parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Load data
    geom = GEOMDatamodule(
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        split_ratio=cfg.split_ratio,
        num_workers=cfg.num_workers,
        tol=cfg.tol,
    )

    print("Datamodule loaded.")

    # Initialize and load model
    ddpm = LitEnEquivariantDDPM(
        config=cfg,
        loss_type=cfg.loss_type,
        lr=cfg.lr,
        ema_decay=cfg.ema_decay,
        clip_grad_norm=cfg.clip_grad_norm,
        n_visualize_samples=cfg.n_visualize_samples,
        n_sample_metric_batches=cfg.n_sample_metric_batches,
        guidance_scales=cfg.guidance_scales,
    )

    print("Model loaded.")

    if cfg.wandb:
        project = "train_e(n)_equiv_ddpm" + ("_debug" if cfg.debug else "")
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
            "limit_train_batches": 5,
            "limit_val_batches": 5,
            "limit_test_batches": 5,
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
    pydantic_cli.run_and_exit(TrainEnEquivariantDDPMConfig, train_ddpm)
