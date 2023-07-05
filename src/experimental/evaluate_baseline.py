from src.datamodule import ConformerDatamodule
from src.metrics import evaluate_prediction
import torch
import pathlib
import pydantic
import pydantic_cli
import json
import tqdm
import numpy as np
from torch.utils.data import Subset
from src import chem, utils, baseline
from typing import List

class EvaluateBaselineConfig(pydantic.BaseModel):

    dataset: str = 'qm9'
    split: str = 'test'
    save_dir: str = '/local-scratch/baseline'
    chunk_id: int = 0
    num_chunks: int = 100
    save_samples_and_examples: bool = False
    only_carbon_cond: bool = False
    pdropout_cond: List[float] = (0.0, 0.0)
    save_samples_and_examples: bool = False

    ngen: int = 20 # number of generations
    popsize: int = 20_000 # population size
    num_samples: int = 10 # number of samples to generate

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def evaluate_baseline(config: EvaluateBaselineConfig):
    cfg = config

    data = ConformerDatamodule(
        dataset=cfg.dataset,
        seed=100,
        split_ratio=(0.8, 0.1, 0.1),
        tol=-1.0,
        batch_size=0,
        num_workers=0,
        only_lowest_energy_conformers=True,
    )

    dataset = data.datasets[cfg.split]
    chunk_indices = np.array_split(range(len(dataset)), cfg.num_chunks)[cfg.chunk_id]

    chunk = Subset(dataset, chunk_indices)

    # Creating saving directory
    save_dir = pathlib.Path(cfg.save_dir) / cfg.dataset / str(cfg.pdropout_cond[1]) / f"chunk-{cfg.chunk_id}"
    save_dir.mkdir(exist_ok=True, parents=True)
    progress_path = save_dir / "progress.pt"
    results_path = save_dir / "results.pt"

    with open(save_dir / "config.json", "w+") as f:
        config_export = cfg.dict()
        json.dump(config_export, f, indent=2)

    if results_path.exists() and results_path.stat().st_size > 0:
        print("Results already exist, skipping")
        return 0

    if progress_path.exists():
        evaluated_mols, results = torch.load(progress_path, map_location="cpu")
    else:
        evaluated_mols, results = -1, {}

    # Evaluate
    for idx, M in enumerate(tqdm.tqdm(chunk, desc="Evaluating")):
        if idx <= evaluated_mols:
            continue

        # Dropout conditioning labels
        M = utils.dropout_unsigned_coords(M, prange=cfg.pdropout_cond)
        if cfg.only_carbon_cond:
            nonC_mask = (M.atom_nums != 6)
            M = utils.dropout_unsigned_coords(M, dropout_mask=nonC_mask)
        
        M_samples = baseline.run_baseline(M, ngen=cfg.ngen, dataset=cfg.dataset, verbose=False, popsize=cfg.popsize, num_samples=cfg.num_samples)

        # Compute metrics
        M_true = M.replace(graph=None)
        log = {
            "mol": M_true if cfg.save_samples_and_examples else None,
            "preds": [],
        }

        for M_pred in M_samples:
            assert M_pred.id_as_int == M_true.id_as_int

            metrics, M_aligned = evaluate_prediction(
                M_pred=M_pred,
                M_true=M_true,
                keep_coords_pred=True,
                return_aligned_mol=True,
            )
            metrics["coords"] = M_aligned.coords if cfg.save_samples_and_examples else None
            log["preds"].append(metrics)

        results[M_true.id_as_int] = log

        torch.save((idx, results), progress_path)

    torch.save(results, results_path)
    progress_path.unlink(missing_ok=True)
    print("Finished!")

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(EvaluateBaselineConfig, evaluate_baseline)
