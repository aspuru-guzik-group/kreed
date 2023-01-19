import collections
import statistics

import dgl
import pytorch_lightning as pl
import spyrmsd.rmsd
import torch
import torch_ema
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

from src.diffusion.configs import EquivariantDDPMConfig
from src.diffusion.ddpm import EnEquivariantDDPM, RefEquivariantDDPM
from src.modules import EGNNDynamics, KraitchmanClassifier
from src.visualize import html_render_molecule, html_render_trajectory
from src.xyz2mol import xyz2mol


class LitEquivariantDDPM(pl.LightningModule):

    def __init__(
        self,
        config: EquivariantDDPMConfig,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        dynamics = EGNNDynamics(
            d_atom_vocab=config.d_egnn_atom_vocab,
            d_hidden=config.d_egnn_hidden,
            n_layers=config.n_egnn_layers,
            equivariance=config.equivariance,
        )

        if config.equivariance == "e3":
            if config.clf:
                classifier = KraitchmanClassifier(scale=config.clf_std, stable=config.clf_stable_pi)
            else:
                classifier = None

            self.edm = EnEquivariantDDPM(
                dynamics=dynamics,
                classifier=classifier,
                timesteps=config.timesteps,
                noise_shape=config.noise_shape,
                noise_precision=config.noise_precision,
                loss_type=config.loss_type,
            )

        elif config.equivariance == "reflection":
            assert not config.clf

            self.edm = RefEquivariantDDPM(
                dynamics=dynamics,
                timesteps=config.timesteps,
                noise_shape=config.noise_shape,
                noise_precision=config.noise_precision,
                loss_type=config.loss_type,
                loss_weight=config.loss_weight,
            )

        trainable_params = [p for p in self.edm.parameters() if p.requires_grad]
        self.ema = torch_ema.ExponentialMovingAverage(trainable_params, decay=config.ema_decay)
        self.ema_moved_to_device = False

        self.grad_norm_queue = collections.deque([3000, 3000], maxlen=50)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.edm.parameters(), lr=self.config.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if not self.ema_moved_to_device:
            self.ema.to(self.device)
            self.ema_moved_to_device = True
        self.ema.update()

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None):
        if not self.config.clip_grad_norm:
            return

        max_norm = 1.5 * statistics.mean(self.grad_norm_queue) + 2 * statistics.stdev(self.grad_norm_queue)
        self.log("max_grad_norm", max_norm)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.edm.parameters(), max_norm=max_norm, norm_type=2.0, error_if_nonfinite=True).item()
        grad_norm = min(grad_norm, max_norm)
        self.grad_norm_queue.append(grad_norm)

    def training_step(self, batch, batch_idx):
        nll = self._step(batch, split="train")
        if batch_idx < self.config.n_sample_metric_batches:
            n_visualize = self.config.n_visualize_samples if (batch_idx == 0) else 0
            self._evaluate_samples(batch, split="train", n_visualize=n_visualize)
        return nll

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, split="val")
            if batch_idx < self.config.n_sample_metric_batches:
                n_visualize = self.config.n_visualize_samples if (batch_idx == 0) else 0
                self._evaluate_samples(batch, split="val", n_visualize=n_visualize)
            return nll

    def test_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            nll = self._step(batch, split="test")
            n_visualize = self.config.n_visualize_samples if (batch_idx == 0) else 0
            self._evaluate_samples(batch, split="test", n_visualize=n_visualize)
            return nll

    def _step(self, G, split):
        nll = self.edm(G).mean()
        self.log(f"{split}/nll", nll, batch_size=G.batch_size)
        return nll

    def _evaluate_samples(self, G, split, n_visualize):
        for s in self.config.guidance_scales:
            self._evaluate_guided_samples(G=G, split=split, n_visualize=n_visualize, scale=s)

    def _evaluate_guided_samples(self, G, split, n_visualize, scale):
        folder = f"{split}/samples_scale={scale}"
        G_sample, frames = self.edm.sample_p_G0(G_init=G, guidance_scale=scale, keep_frames=range(self.config.timesteps, 0, -1))

        rmsd = 0.0
        stability = 0.0

        for i, (G_true, G_pred) in enumerate(zip(dgl.unbatch(G), dgl.unbatch(G_sample))):
            geom_id = G_true.ndata["id"][0].item()
            atom_nums = G_true.ndata["atom_nums"].cpu().numpy()
            coords_true = G_true.ndata["xyz"].cpu().numpy()
            coords_pred = G_pred.ndata["xyz"].cpu().numpy()

            if i < n_visualize:
                if isinstance(self.logger, WandbLogger):
                    wandb.log({
                        f"{folder}/true_{i}": wandb.Html(html_render_molecule(geom_id, atom_nums, coords_true)),
                        f"{folder}/pred_{i}": wandb.Html(html_render_molecule(geom_id, atom_nums, coords_pred)),
                        "epoch": self.current_epoch,
                    })

                    atom_nums_list = []
                    coords_list = []
                    for t, batch in frames.items():
                        graph = dgl.unbatch(batch)[i]
                        atom_nums_list.append(graph.ndata["atom_nums"].cpu().numpy())
                        coords_list.append(graph.ndata["xyz"].cpu().numpy())

                    wandb.log({
                        f"{folder}/anim_pred_{i}": wandb.Html(html_render_trajectory(geom_id, atom_nums_list, coords_list)),
                        "epoch": self.current_epoch,
                    })

            # Compute sample metrics

            # take min(left, right) enantiomer
            rmsd1 = spyrmsd.rmsd.rmsd(
                coords1=coords_true,
                coords2=coords_pred,
                atomicn1=atom_nums,
                atomicn2=atom_nums,
                minimize=True
            )

            coords_pred[:, 0] *= -1  # flip x coordinates for other enantiomer

            rmsd2 = spyrmsd.rmsd.rmsd(
                coords1=coords_true,
                coords2=coords_pred,
                atomicn1=atom_nums,
                atomicn2=atom_nums,
                minimize=True
            )

            rmsd += min(rmsd1, rmsd2)

            try:
                mols = xyz2mol(
                    atoms=atom_nums.tolist(),
                    coordinates=coords_pred.tolist(),
                    embed_chiral=False,
                )
            except ValueError:
                mols = False

            stability += (1.0 if mols else 0.0)

        rmsd = rmsd / G.batch_size
        stability = stability / G.batch_size

        self.log(f"{folder}/rmsd", rmsd, batch_size=G.batch_size)
        self.log(f"{folder}/stability", stability, batch_size=G.batch_size)
