import collections
import statistics

import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

from src import utils
from src.diffusion.ddpm import EquivariantDDPM, EquivariantDDPMConfig
from src.metrics import evaluate_prediction
from src.modules import EMA
from src.visualize import html_render_molecule, html_render_trajectory


class LitEquivariantDDPM(pl.LightningModule):

    def __init__(
        self,
        config: EquivariantDDPMConfig,
        lr, wd,
        clip_grad_norm,
        ema_decay,
        puncond,
        pdropout_cond,
        check_samples_every_n_epochs,
        samples_visualize_n_mols,
        samples_assess_n_batches,
        samples_render_every_n_frames,
        distributed,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.edm = EquivariantDDPM(config=config)
        self.ema = EMA(self.edm, beta=ema_decay)

        grad_norm_queue = torch.full([50], fill_value=3000, dtype=torch.float)
        self.register_buffer("grad_norm_queue", grad_norm_queue)

    # Reference: https://github.com/Tony-Y/pytorch_warmup
    def linear_warmup(self, step):
        return min(step, 2000) / 2000

    def configure_optimizers(self):
        params = []
        params_no_wd = []

        for name, p in self.edm.named_parameters():
            *attrs, name = name.split(".")

            # Get parent module
            parent = self.edm
            for k in attrs:
                parent = getattr(parent, k)

            # Sort parameters
            if isinstance(parent, (nn.Embedding, nn.LayerNorm)) or (name == "bias"):
                params_no_wd.append(p)
            else:
                params.append(p)

        optimizer = torch.optim.AdamW(
            params=[
                {"params": params, "lr": self.hparams.lr, "weight_decay": self.hparams.wd},
                {"params": params_no_wd, "lr": self.hparams.lr, "weight_decay": 0.0},
            ],
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=[self.linear_warmup, self.linear_warmup],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step",
            }
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if not self.hparams.clip_grad_norm:
            return

        max_norm = (1.5 * self.grad_norm_queue.mean()) + (2 * self.grad_norm_queue.std(unbiased=False))
        self.log("max_grad_norm", max_norm.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.edm.parameters(),
            max_norm=max_norm.item(),
            norm_type=2.0,
            error_if_nonfinite=True
        )

        grad_norm = min(grad_norm, max_norm)
        self.grad_norm_queue[self.global_step % self.grad_norm_queue.shape[0]] = grad_norm

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.edm)

    def training_step(self, batch, batch_idx):
        hp = self.hparams
        M = batch

        # Dropout conditioning labels
        M = utils.dropout_unsigned_coords(M, prange=hp.pdropout_cond)

        # Compute loss
        loss = self.edm.simple_losses(M, puncond=hp.puncond).mean()
        self.log(f"train/loss", loss, batch_size=M.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        hp = self.hparams
        M = batch

        # Dropout conditioning labels
        nonC_mask = (M.atom_nums != 6)
        M = utils.dropout_unsigned_coords(M, dropout_mask=nonC_mask)
        M = utils.dropout_unsigned_coords(M, prange=0.1)

        # Visualize and assess some samples
        if (
            ((self.current_epoch + 1) % hp.check_samples_every_n_epochs == 0)
            and (batch_idx < hp.samples_assess_n_batches)
        ):
            n = hp.samples_visualize_n_mols if (batch_idx == 0) else 0
            self._assess_and_visualize_samples(M=M, split="val", n_visualize=n)

        # Compute NLL
        nll = self.ema.ema_model.nlls(M).mean()
        self.log(f"val/nll", nll, batch_size=M.batch_size, sync_dist=hp.distributed)
        return nll

    @torch.no_grad()
    def _assess_and_visualize_samples(self, M, split, n_visualize):
        hp = self.hparams

        T = self.config.timesteps
        keep_frames = list(reversed(range(-1, T + 1, hp.samples_render_every_n_frames)))
        M_preds, frames = self.ema.ema_model.sample(M=M, keep_frames=set(keep_frames))

        M_trues = M.cpu().unbatch()
        M_preds = M_preds.cpu().unbatch()
        frames = {step: m.unbatch() for step, m in frames.items()}

        metrics = collections.defaultdict(list)

        for i in tqdm.trange(M.batch_size, desc=f"Evaluating {split} samples", leave=False):
            sample_metrics, M_aligned = evaluate_prediction(M_pred=M_preds[i], M_true=M_trues[i], return_aligned_mol=True)
            for k, v in sample_metrics.items():
                metrics[k].append(v)

            if (i >= n_visualize) or (not isinstance(self.logger, WandbLogger)):
                continue
            if self.global_rank != 0:
                continue

            M_pred_traj = [frames[step][i] for step in tqdm.tqdm(keep_frames + [-1], desc=f"Rendering trajectory {i}", leave=False)]
            M_pred_traj = [m.replace(coords=m.coords) for m in M_pred_traj]

            wandb.log({
                f"{split}_samples/true_{i}": wandb.Html(html_render_molecule(M_trues[i])),
                f"{split}_samples/pred_{i}": wandb.Html(html_render_molecule(M_aligned)),
                f"{split}_samples/anim_pred_{i}": wandb.Html(html_render_trajectory(M_pred_traj)),
                "epoch": self.current_epoch,
            })

        metrics = {f"{split}/{k}": statistics.mean(vs) for k, vs in metrics.items()}
        self.log_dict(metrics, batch_size=M.batch_size, sync_dist=hp.distributed)
