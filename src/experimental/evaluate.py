import copy
import json
import pathlib
from typing import List

import numpy as np
import pydantic
import pydantic_cli
import torch
import tqdm
from lightning_fabric.utilities.seed import pl_worker_init_function
from torch.utils.data import DataLoader, Subset

from src import chem, utils
from src.metrics import evaluate_prediction
from src.datamodule import ConformerDatamodule
from src.diffusion import LitEquivariantDDPM
from src.experimental.train import TrainEquivariantDDPMConfig

assert TrainEquivariantDDPMConfig, "Needed for checkpoint loading!"

torch.set_float32_matmul_precision("medium")


class EvaluateEquivariantDDPMConfig(pydantic.BaseModel):
    """Configuration object for evaluating the DDPM."""

    debug: bool = False

    accelerator: str = "gpu"

    # =================
    # Datamodule Fields
    # =================

    split: str = "test"
    batch_size: int = 500
    num_workers: int = 8

    chunk_id: int = 0
    num_chunks: int = 100

    # ===================
    # Experimental Fields
    # ===================

    checkpoint_path: str
    sample_n_candidates_per_example: int = 10

    only_carbon_cond: bool = False
    pdropout_cond: List[float] = (0.0, 0.0)

    # ==============
    # Logging Fields
    # ==============

    save_dir: str = "results/debug"
    save_samples_and_examples: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def evaluate_ddpm(config: EvaluateEquivariantDDPMConfig):
    cfg = config
    assert cfg.batch_size % cfg.sample_n_candidates_per_example == 0

    device = torch.device("cuda" if ((cfg.accelerator == "gpu") and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # Load checkpoint
    ckpt_path = pathlib.Path(cfg.checkpoint_path)
    assert ckpt_path.is_file()
    ddpm = LitEquivariantDDPM.load_from_checkpoint(ckpt_path, map_location=device)
    ddpm = ddpm.ema.ema_model
    ddpm.eval()

    if cfg.debug:
        ddpm.T = 3

    # Load data; keep same datamodule config from training (!!!)
    train_config = ddpm.config
    data = ConformerDatamodule(
        dataset=train_config.dataset,
        seed=train_config.seed,
        split_ratio=train_config.split_ratio,
        tol=train_config.tol,
        batch_size=0,
        num_workers=0,
        only_lowest_energy_conformers=True,
    )

    dataset = data.datasets[cfg.split]
    chunk_indices = np.array_split(range(len(dataset)), cfg.num_chunks)[cfg.chunk_id]

    loader = DataLoader(
        dataset=Subset(dataset, chunk_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        worker_init_fn=pl_worker_init_function,
        collate_fn=chem.Molecule.collate,
        pin_memory=True,
    )

    # Creating saving directory
    save_dir = pathlib.Path(cfg.save_dir) / f"chunk-{cfg.chunk_id}"
    save_dir.mkdir(exist_ok=True, parents=True)
    progress_path = save_dir / "progress.pt"
    results_path = save_dir / "results.pt"

    with open(save_dir / "config.json", "w+") as f:
        config_export = cfg.dict()
        config_export["train_config"] = train_config.dict()
        json.dump(config_export, f, indent=2)

    if progress_path.exists():
        evaluated_batches, results = torch.load(progress_path, map_location="cpu")
    else:
        evaluated_batches, results = -1, {}

    # Evaluate
    for batch_idx, M in enumerate(tqdm.tqdm(loader, desc="Evaluating")):
        if batch_idx <= evaluated_batches:
            continue
        M = data.transfer_batch_to_device(M, device, 0)

        # Dropout conditioning labels
        M = utils.dropout_unsigned_coords(M, prange=cfg.pdropout_cond)
        if cfg.only_carbon_cond:
            nonC_mask = (M.atom_nums != 6)
            M = utils.dropout_unsigned_coords(M, dropout_mask=nonC_mask)

        # Repeat entries in batch
        M_repeated = []
        for m in M.unbatch():
            M_repeated.extend([copy.deepcopy(m) for _ in range(cfg.sample_n_candidates_per_example)])
        M_repeated = chem.Molecule.collate(M_repeated)

        # Sample
        M_samples = ddpm.sample(M_repeated).cpu().unbatch()

        # Compute metrics
        for i, M_true in enumerate(M.cpu().unbatch()):

            log = {
                "mol": M_true.replace(graph=None) if cfg.save_samples_and_examples else None,
                "preds": [],
            }

            start = cfg.sample_n_candidates_per_example * i
            end = cfg.sample_n_candidates_per_example * (i + 1)
            for M_pred in M_samples[start:end]:
                assert M_pred.id_as_int == M_true.id_as_int

                metrics, M_aligned = evaluate_prediction(
                    M_pred=M_pred,
                    M_true=M_true,
                    return_aligned_mol=True,
                )
                metrics["coords"] = M_aligned.coords if cfg.save_samples_and_examples else None
                log["preds"].append(metrics)

            results[M_true.id_as_int] = log

        torch.save((batch_idx, results), progress_path)

    torch.save(results, results_path)
    progress_path.unlink(missing_ok=True)
    print("Finished!")

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(EvaluateEquivariantDDPMConfig, evaluate_ddpm)
