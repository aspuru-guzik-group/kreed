import argparse
import pathlib

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import UnsignedCoordinateDatamodule
from src.experiments.signing.model import PLCoordinateSignPredictor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1244)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tol", type=int, default=-1.0)

    parser.add_argument("--d_embed", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=5)

    parser.add_argument("--epochs", type=float, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    # seed
    pl.seed_everything(args.seed)

    # setup directories
    run_name = f"geom_coord_sign_prediction"

    root = pathlib.Path(__file__).parents[3]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    # load data
    datamodule = UnsignedCoordinateDatamodule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tol=args.tol,
    )

    # initialize and load model
    pl_model = PLCoordinateSignPredictor(
        d_embed=args.d_embed,
        d_vocab=datamodule.d_vocab,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr,
    )

    logger = WandbLogger(project=run_name, log_model=False, save_dir=str(log_dir))
    wandb.config.update(vars(args))

    trainer = pl.Trainer(
        enable_checkpointing=False,  # TODO: disable for prototyping
        logger=logger,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
        deterministic=True,
        overfit_batches=10
    )

    trainer.fit(model=pl_model, datamodule=datamodule)
    trainer.validate(model=pl_model, datamodule=datamodule)
    trainer.test(model=pl_model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
