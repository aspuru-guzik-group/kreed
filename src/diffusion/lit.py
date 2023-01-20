import collections
import statistics
from typing import List

import dgl
import pytorch_lightning as pl
import spyrmsd.rmsd
import torch
import torch_ema
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

from src.diffusion.ddpm import EnEquivariantDDPM, EquivariantDDPMConfig
from src.visualize import html_render_molecule, html_render_trajectory
from src.xyz2mol import xyz2mol

from rdkit import Chem
from tqdm import tqdm

class LitEquivariantDDPMConfig(EquivariantDDPMConfig):
    """Configuration object for the Pytorch-Lightning DDPM wrapper."""

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = 500
    lr: float = 1e-4

    ema_decay: float = 0.9999
    clip_grad_norm: bool = True

    # ================
    # Sampling Fields
    # ================

    n_visualize_samples: int = 3
    n_sample_metric_batches: int = 1

    guidance_scales: List[float] = (0,)


class LitEquivariantDDPM(pl.LightningModule):

    def __init__(self, config: LitEquivariantDDPMConfig):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.edm = EnEquivariantDDPM(config=config)

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
        return self._step(batch, split="train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            return self._step(batch, split="val", batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        with self.ema.average_parameters():
            return self._step(batch, split="test", batch_idx=batch_idx)

    def _step(self, G, split, batch_idx):
        if split == "train":
            loss = self.edm.simple_losses(G).mean()
            self.log(f"{split}/loss", loss, batch_size=G.batch_size)
        else:
            loss = self.edm.nlls(G).mean()
            self.log(f"{split}/nll", loss, batch_size=G.batch_size)

        cfg = self.config
        if batch_idx < cfg.n_sample_metric_batches:
            n = cfg.n_visualize_samples if (batch_idx == 0) else 0
            for scale in cfg.guidance_scales:
                self._evaluate_guided_samples(G=G, split=split, n_visualize=n, scale=scale)

        return loss

    def _evaluate_guided_samples(self, G, split, n_visualize, scale):
        folder = f"{split}/samples_scale={scale}"

        T = self.config.timesteps
        keep_frames = set(range(-1, T + 1))
        G_sample, frames = self.edm.sample_p_G(G_init=G, guidance_scale=scale, keep_frames=keep_frames)

        rmsd = 0.0
        stability = 0.0
        correctness = 0.0

        for i, (G_true, G_pred) in tqdm(enumerate(zip(dgl.unbatch(G), dgl.unbatch(G_sample))), total=G.batch_size, desc=f"Evaluating {folder}", leave=False):
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

                    trajectory = []
                    for step in tqdm(reversed(range(-1, T + 1)), desc=f"Rendering trajectory {i}", leave=False, total=T+2):
                        graph = dgl.unbatch(frames[step])[i]
                        trajectory.append(graph.ndata["xyz"].cpu().numpy())

                    wandb.log({
                        f"{folder}/anim_pred_{i}": wandb.Html(html_render_trajectory(geom_id, atom_nums, trajectory)),
                        "epoch": self.current_epoch,
                    })

            # Compute sample metrics

            carbon_mask = (atom_nums == 6)
            C_coords_true = coords_true[carbon_mask]
            C_coords_pred = coords_pred[carbon_mask]
            C_atomic_nums = atom_nums[carbon_mask]
            
            # take min(left, right) enantiomer
            rmsd1 = spyrmsd.rmsd.rmsd(
                coords1=C_coords_true,
                coords2=C_coords_pred,
                atomicn1=C_atomic_nums,
                atomicn2=C_atomic_nums,
                minimize=True
            )

            C_coords_pred[:, 0] *= -1  # flip x coordinates for other enantiomer

            rmsd2 = spyrmsd.rmsd.rmsd(
                coords1=C_coords_true,
                coords2=C_coords_pred,
                atomicn1=C_atomic_nums,
                atomicn2=C_atomic_nums,
                minimize=True
            )

            rmsd += min(rmsd1, rmsd2)

            try:
                pred_mols = xyz2mol(
                    atoms=atom_nums.tolist(),
                    coordinates=coords_pred.tolist(),
                    embed_chiral=False,
                )
            except ValueError:
                pred_mols = False

            stability += (1.0 if pred_mols else 0.0)

            if pred_mols:
                try:
                    true_mols = xyz2mol(
                        atoms=atom_nums.tolist(),
                        coordinates=coords_true.tolist(),
                        embed_chiral=False,
                    )
                except ValueError:
                    true_mols = False
                
                if true_mols:
                    true_smiles = Chem.MolToSmiles(true_mols[0])
                    pred_smiles = Chem.MolToSmiles(pred_mols[0])
                    correctness += (1.0 if true_smiles == pred_smiles else 0.0)

        rmsd = rmsd / (G.batch_size)
        stability = stability / G.batch_size
        correctness = correctness / G.batch_size

        self.log(f"{folder}/carbon_rmsd", rmsd, batch_size=G.batch_size)
        self.log(f"{folder}/stability", stability, batch_size=G.batch_size)
        self.log(f"{folder}/correctness", correctness, batch_size=G.batch_size)
