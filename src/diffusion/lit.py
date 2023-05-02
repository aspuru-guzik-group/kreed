import collections
import statistics

import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

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

        self.grad_norm_queue = collections.deque([3000, 3000], maxlen=50)

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

        max_norm = 1.5 * statistics.mean(self.grad_norm_queue) + 2 * statistics.stdev(self.grad_norm_queue)
        self.log("max_grad_norm", max_norm)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.edm.parameters(), max_norm=max_norm, norm_type=2.0, error_if_nonfinite=True).item()
        grad_norm = min(grad_norm, max_norm)
        self.grad_norm_queue.append(grad_norm)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.edm)

    def training_step(self, batch, batch_idx):
        return self._step(batch, split="train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, split="val", batch_idx=batch_idx)

    def _step(self, M, split, batch_idx):
        hp = self.hparams

        # Dropout on conditioning labels
        if hp.pdropout_cond > 0:
            dropout_mask = (torch.rand_like(M.coords) < hp.pdropout_cond)
            cond_mask = M.cond_mask & (~dropout_mask)
            M = M.replace(cond_mask=cond_mask, cond_labels=torch.where(cond_mask, M.cond_labels, 0.0))

        if split == "train":
            loss = self.edm.simple_losses(M, puncond=hp.puncond).mean()
            self.log(f"{split}/loss", loss, batch_size=M.batch_size)

        else:
            # Visualize and assess some samples
            if (
                ((self.current_epoch + 1) % hp.check_samples_every_n_epochs == 0)
                and (batch_idx < hp.samples_assess_n_batches)
            ):
                n = hp.samples_visualize_n_mols if (batch_idx == 0) else 0
                self._assess_and_visualize_samples(M=M, split=split, n_visualize=n)

            loss = self.ema.ema_model.nlls(M).mean()
            self.log(f"{split}/nll", loss, batch_size=M.batch_size, sync_dist=hp.distributed)

        return loss

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
            sample_metrics = evaluate_prediction(M_pred=M_preds[i], M_true=M_trues[i])
            transform = sample_metrics.pop("transform")
            for k, v in sample_metrics.items():
                metrics[k].append(v)

            if (i >= n_visualize) or (not isinstance(self.logger, WandbLogger)):
                continue
            if self.global_rank != 0:
                continue

            M_pred_traj = [frames[step][i] for step in tqdm.tqdm(keep_frames + [-1], desc=f"Rendering trajectory {i}", leave=False)]
            M_pred_traj = [m.replace(coords=m.coords).transform(transform) for m in M_pred_traj]

            wandb.log({
                f"{split}_samples/true_{i}": wandb.Html(html_render_molecule(M_trues[i])),
                f"{split}_samples/pred_{i}": wandb.Html(html_render_molecule(M_preds[i].transform(transform))),
                f"{split}_samples/anim_pred_{i}": wandb.Html(html_render_trajectory(M_pred_traj)),
                "epoch": self.current_epoch,
            })

        metrics = {f"{split}/{k}": statistics.mean(vs) for k, vs in metrics.items()}
        self.log_dict(metrics, batch_size=M.batch_size, sync_dist=hp.distributed)
