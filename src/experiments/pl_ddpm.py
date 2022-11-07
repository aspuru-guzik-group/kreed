import collections
import statistics

import pytorch_lightning as pl
import torch
import torch_ema

from src.diffusion import EGNNDynamics, EnEquivariantDiffusionModel

from src.datamodules.geom import GEOM_ATOMS
import dgl

import wandb
from src.utils import make_html

# from ase import Atoms
# from ase.io import write
# from wandb import Molecule

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
        n_sample_batches=20,
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

        self.on_correct_device = False

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.edm.parameters(), lr=self.hparams.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if not self.on_correct_device:
            self.ema.to(self.device)
            self.on_correct_device = True
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
        if batch_idx < self.hparams.n_sample_batches:
            self._sample(batch, "train")
        return nll

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, "val")
            if batch_idx < self.hparams.n_sample_batches:
                self._sample(batch, "val")
            return nll

    def test_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, "test")
            self._sample(batch, "test")
            return nll

    def _step(self, G, split):
        nll = self.edm(G).mean()
        self.log(f"{split}_nll", nll, batch_size=G.batch_size)
        return nll

    def _sample(self, G_init, split):
        G_sample = self.edm.sample_p_G0(G_init=G_init)
        mol = dgl.unbatch(G_sample)[0]
        html = make_html(mol)
        wandb.log({f"{split}_molecule": wandb.Html(html)})

        
        # atoms = Atoms(numbers=Z, positions=XYZ)
        # write('atoms.pdb', atoms)
        # wandb.log({f"{split}_atoms":Molecule('atoms.pdb')})