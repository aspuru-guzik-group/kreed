import collections
import statistics

import dgl
import pytorch_lightning as pl
import spyrmsd.rmsd
import torch
import torch_ema
import wandb
import xyz2mol
from rdkit import Chem

from src.datamodules.geom import GEOM_ATOMS
from src.diffusion import EGNNDynamics, EnEquivariantDiffusionModel
from src.visualize import html_render


class PlEnEquivariantDiffusionModel(pl.LightningModule):

    def __init__(
        self,
        d_egnn_atom_vocab=16,
        d_egnn_hidden=256,
        n_egnn_layers=4,
        timesteps=1000,
        noise_shape="polynomial_2",
        noise_precision=0.08,
        loss_type="L2",
        lr=1e-4,
        ema_decay=0.999,
        clip_grad_norm=True,
        n_visualize_samples=3,
        n_sample_metric_batches=20,
    ):
        super().__init__()

        self.save_hyperparameters()

        dynamics = EGNNDynamics(
            d_atom_vocab=d_egnn_atom_vocab,
            d_hidden=d_egnn_hidden,
            n_layers=n_egnn_layers,
        )

        self.edm = EnEquivariantDiffusionModel(
            dynamics=dynamics,
            timesteps=timesteps,
            noise_shape=noise_shape,
            noise_precision=noise_precision,
            loss_type=loss_type
        )

        self.ema = torch_ema.ExponentialMovingAverage(self.edm.parameters(), decay=ema_decay)

        self.grad_norm_queue = collections.deque([3000, 3000], maxlen=50)

    def setup(self, stage):
        self.ema.to(self.device)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.edm.parameters(), lr=self.hparams.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None):
        if not self.hparams.clip_grad_norm:
            return

        max_norm = 1.5 * statistics.mean(self.grad_norm_queue) + 2 * statistics.stdev(self.grad_norm_queue)
        self.log("max_grad_norm", max_norm)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.edm.parameters(), max_norm=max_norm, norm_type=2.0, error_if_nonfinite=True).item()
        grad_norm = min(grad_norm, max_norm)
        self.grad_norm_queue.append(grad_norm)

    def training_step(self, batch, batch_idx):
        nll = self._step(batch, "train")
        if batch_idx < self.hparams.n_sample_metric_batches:
            self._visualize_and_check_samples(batch, "train", n_visualize=0)
        return nll

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, "val")
            if batch_idx < self.hparams.n_sample_metric_batches:
                n_visualize = self.hparams.n_visualize_samples if (batch_idx == 0) else 0
                self._visualize_and_check_samples(batch, "val", n_visualize=n_visualize)
            return nll

    def test_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, "test")
            n_visualize = self.hparams.n_visualize_samples if (batch_idx == 0) else 0
            self._visualize_and_check_samples(batch, "test", n_visualize=n_visualize)
            return nll

    def _step(self, G, split):
        nll = self.edm(G).mean()
        self.log(f"{split}/nll", nll, batch_size=G.batch_size)
        return nll

    def _visualize_and_check_samples(self, G, split, n_visualize):
        G_sample = self.edm.sample_p_G0(G_init=G)

        rmsd = 0.0
        stability = 0.0

        for i, (G_true, G_pred) in enumerate(zip(dgl.unbatch(G), dgl.unbatch(G_sample))):
            geom_id = G_true.ndata["id"][0].item()
            atom_nums = GEOM_ATOMS[G_true.ndata["atom_nums"]].cpu().numpy()
            coords_true = G_true.ndata["xyz"].cpu().numpy()
            coords_pred = G_pred.ndata["xyz"].cpu().numpy()

            if i < n_visualize:
                wandb.log({
                    f"{split}/samples/true_{i}": wandb.Html(html_render(geom_id, atom_nums, coords_true)),
                    f"{split}/samples/pred_{i}": wandb.Html(html_render(geom_id, atom_nums, coords_pred)),
                    "epoch": self.current_epoch,
                })

            # Compute sample metrics
            rmsd += spyrmsd.rmsd.rmsd(
                coords1=coords_true,
                coords2=coords_pred,
                atomicn1=atom_nums,
                atomicn2=atom_nums,
                minimize=True
            )

            mols = xyz2mol.xyz2mol(
                atoms=atom_nums.tolist(),
                coordinates=coords_pred.tolist(),
                embed_chiral=False,
            )

            stability += (1.0 if mols else 0.0)

        rmsd = rmsd / G.batch_size
        stability = stability / G.batch_size

        self.log(f"{split}/rmsd", rmsd, batch_size=G.batch_size)
        self.log(f"{split}/stability", stability, batch_size=G.batch_size)

        # from ase import Atoms
        # from ase.io import write
        # from wandb import Molecule
        # atoms = Atoms(numbers=Z, positions=XYZ)
        # write('atoms.pdb', atoms)
        # wandb.log({f"{split}_atoms":Molecule('atoms.pdb')})
