from typing import List

import pydantic
import pydantic_cli


class EnEquivariantDDPMConfig(pydantic.BaseModel):
    """Configuration object for the DDPM."""

    d_egnn_atom_vocab: int = 16
    d_egnn_hidden: int = 256
    n_egnn_layers: int = 4

    timesteps: int = 1000
    noise_shape: str = "polynomial_2"
    noise_precision: float = 0.08


class TrainEnEquivariantDDPMConfig(EnEquivariantDDPMConfig):
    """Configuration object for training the DDPM."""

    seed: int = 100
    debug: bool = False

    accelerator: str = "cpu"
    devices: int = 1

    # =================
    # Datamodule Fields
    # =================

    batch_size: int = 64
    split_ratio: List[float] = (0.8, 0.1, 0.1)
    num_workers: int = 0
    tol: float = -1.0

    # ===============
    # Training Fields
    # ===============

    loss_type: str = "L2"

    max_epochs: int = 500
    lr: float = 1e-4

    ema_decay: float = 0.999
    clip_grad_norm: bool = True

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    checkpoint: bool = False

    log_every_n_steps: int = 10
    progress_bar: bool = False

    n_visualize_samples: int = 3
    n_sample_metric_batches: int = 20

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")
