import collections
import statistics
from typing import List

import dgl
import pytorch_lightning as pl
import torch_ema
import torch
import tqdm
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

from src.diffusion.ddpm import EquivariantDDPM, EquivariantDDPMConfig
from src.evaluate import evaluate
from src.visualize import html_render_molecule, html_render_trajectory


class LitEquivariantDDPM(pl.LightningModule):

    def __init__(
        self,
        config: EquivariantDDPMConfig,
        lr,
        clip_grad_norm,
        puncond,
        check_samples_every_n_epoch,
        samples_visualize_n_mols,
        samples_assess_n_batches,
        samples_render_every_n_frames,
        n_eval_samples,
        distributed,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.edm = EquivariantDDPM(config=config)
        self.grad_norm_queue = collections.deque([3000, 3000], maxlen=50)

        trainable_params = [p for p in self.edm.parameters() if p.requires_grad]
        self.ema = torch_ema.ExponentialMovingAverage(trainable_params, decay=config.ema_decay)
        self.ema_moved_to_device = False

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.edm.parameters(), lr=self.hparams.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        if not self.ema_moved_to_device:
            self.ema.to(self.device)
            self.ema_moved_to_device = True
        self.ema.update()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if not self.hparams.clip_grad_norm:
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

    # def _step(self, G, split, batch_idx):
    #     hp = self.hparams
    #     if (self.current_epoch % hp.check_samples_every_n_epoch == 0) and (batch_idx < hp.samples_assess_n_batches):
    #         n = hp.samples_visualize_n_mols if (batch_idx == 0) else 0
    #         self._assess_and_visualize_samples(G=G, split=split, n_visualize=n)

    #     if split == "train":
    #         loss = self.edm.simple_losses(G, puncond=hp.puncond).mean()
    #         self.log(f"{split}/loss", loss, batch_size=G.batch_size)
    #     else:
    #         loss = self.edm.nlls(G).mean()
    #         self.log(f"{split}/nll", loss, batch_size=G.batch_size)
    #     return loss
    
    def _step(self, G, split, batch_idx):
        hp = self.hparams
        if (self.current_epoch % hp.check_samples_every_n_epoch == 0) and (batch_idx < hp.samples_assess_n_batches):
            n = hp.samples_visualize_n_mols if (batch_idx == 0) else 0
            small_G = dgl.batch(dgl.unbatch(G)[:hp.n_eval_samples])
            self._assess_and_visualize_samples(G=small_G, split=split, n_visualize=n)

        if split == "train":
            loss = self.edm.simple_losses(G).mean()
            self.log(f"{split}/loss", loss, batch_size=G.batch_size)
        else:
            loss = self.edm.nlls(G).mean()
            self.log(f"{split}/nll", loss, batch_size=G.batch_size)

        return loss

    @torch.no_grad()
    def _assess_and_visualize_samples(self, G, split, n_visualize):
        folder = f"{split}/samples"
        hp = self.hparams

        T = self.config.timesteps
        keep_frames = list(reversed(range(-1, T + 1, hp.samples_render_every_n_frames)))
        G_sample, frames = self.edm.sample_p_G(G_init=G, keep_frames=set(keep_frames))

        frames = {step: dgl.unbatch(g) for step, g in tqdm.tqdm(frames.items(), desc=f"Unbatching {folder}", leave=False)}

        abs_C_rmsds = 0.0
        heavy_rmsds = 0.0
        correctness = 0.0

        for i, (G_true, G_pred) in tqdm.tqdm(enumerate(zip(dgl.unbatch(G), dgl.unbatch(G_sample))), total=G.batch_size, desc=f"Evaluating {folder}", leave=False):
            G_true = G_true.cpu()
            G_pred = G_pred.cpu()

            G_pred.ndata['xyz'] = torch.where(torch.isnan(G_pred.ndata['xyz']), 0.0, G_pred.ndata['xyz'])
            
            abs_C_rmsd, heavy_rmsd, correct, best_flip = evaluate(G_true, G_pred)
            abs_C_rmsds += abs_C_rmsd
            heavy_rmsds += heavy_rmsd
            correctness += correct

            geom_id = G_true.ndata["id"][0].item()
            atom_nums = G_true.ndata["atom_nums"].cpu().numpy()
            coords_true = G_true.ndata["xyz"].cpu().numpy()
            coords_pred = (G_pred.ndata["xyz"].cpu() * best_flip).numpy()

            if (i >= n_visualize) or (not isinstance(self.logger, WandbLogger)):
                continue
            if self.global_rank != 0:
                continue

            wandb.log({
                f"{folder}/true_{i}": wandb.Html(html_render_molecule(geom_id, atom_nums, coords_true)),
                f"{folder}/pred_{i}": wandb.Html(html_render_molecule(geom_id, atom_nums, coords_pred)),
                "epoch": self.current_epoch,
            })
            
            trajectory = []
            for step in tqdm.tqdm(keep_frames + [-1], desc=f"Rendering trajectory {i}", leave=False):
                # + [-1] to render the initial frame
                graph = frames[step][i]
                trajectory.append((graph.ndata["xyz"].cpu() * best_flip).numpy())
            

            wandb.log({
                f"{folder}/anim_pred_{i}": wandb.Html(html_render_trajectory(geom_id, atom_nums, trajectory)),
                "epoch": self.current_epoch,
            })

        abs_C_rmsds /= G.batch_size
        heavy_rmsds /= G.batch_size
        correctness /= G.batch_size
        self.log(f"{folder}/carbon_abs_rmsd", abs_C_rmsds, batch_size=G.batch_size)
        self.log(f"{folder}/heavy_rmsd", heavy_rmsds, batch_size=G.batch_size)
        self.log(f"{folder}/correctness", correctness, batch_size=G.batch_size)

# also remember to comment out EMA in the train.py