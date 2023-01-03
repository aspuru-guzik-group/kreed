from typing import List, Literal

import pydantic
import pydantic_cli


class EnEquivariantDDPMConfig(pydantic.BaseModel):
    """Configuration object for the DDPM."""

    # ===========
    # EGNN Fields
    # ===========

    d_egnn_atom_vocab: int = 16
    d_egnn_hidden: int = 256
    n_egnn_layers: int = 4

    # ===============
    # Schedule Fields
    # ===============

    timesteps: int = 1000
    noise_shape: str = "polynomial_2"
    noise_precision: float = 1e-5

    # =================
    # Classifier Fields
    # =================

    clf: bool = True
    clf_std: float = 1.0
    clf_stable_pi: bool = True


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
    num_workers: int = 4
    tol: float = -1.0

    # ===============
    # Training Fields
    # ===============

    loss_type: Literal["VLB", "L2"] = "L2"

    max_epochs: int = 500
    lr: float = 1e-4

    ema_decay: float = 0.9999
    clip_grad_norm: bool = True

    # ================
    # Sampling Fields
    # ================

    n_visualize_samples: int = 3
    n_sample_metric_batches: int = 20

    guidance_scales: List[float] = (0,)

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    checkpoint: bool = False

    log_every_n_steps: int = 10
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")
