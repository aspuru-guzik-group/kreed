import collections
import statistics

import pytorch_lightning as pl
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
        distributed,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = config

        self.edm = EquivariantDDPM(config=config)
        self.grad_norm_queue = collections.deque([3000, 3000], maxlen=50)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.edm.parameters(), lr=self.hparams.lr)

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
        return self._step(batch, split="val", batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, split="test", batch_idx=batch_idx)

    def _step(self, M, split, batch_idx):
        hp = self.hparams
        if (self.current_epoch % hp.check_samples_every_n_epoch == 0) and (batch_idx < hp.samples_assess_n_batches):
            n = hp.samples_visualize_n_mols if (batch_idx == 0) else 0
            self._assess_and_visualize_samples(M=M, split=split, n_visualize=n)

        if split == "train":
            loss = self.edm.simple_losses(M, puncond=hp.puncond).mean()
            self.log(f"{split}/loss", loss, batch_size=M.batch_size)
        else:
            loss = self.edm.nlls(M).mean()
            self.log(f"{split}/nll", loss, batch_size=M.batch_size)
        return loss

    @torch.no_grad()
    def _assess_and_visualize_samples(self, M, split, n_visualize):
        folder = f"{split}/samples"
        hp = self.hparams

        T = self.config.timesteps
        keep_frames = list(reversed(range(-1, T + 1, hp.samples_render_every_n_frames)))
        M_preds, frames = self.edm.sample(M=M, keep_frames=set(keep_frames))

        M_trues = M.cpu().unbatch()
        M_preds = M_preds.cpu().unbatch()
        frames = {step: m.unbatch() for step, m in frames.items()}

        metrics = collections.defaultdict(list)

        for i in tqdm.trange(M.batch_size, desc=f"Evaluating {folder}", leave=False):
            sample_metrics = evaluate(M_pred=M_preds[i], M_true=M_trues[i])
            for k, v in sample_metrics.items():
                metrics[k].append(v)

            if (i >= n_visualize) or (not isinstance(self.logger, WandbLogger)):
                continue
            if self.global_rank != 0:
                continue

            M_pred_traj = [frames[step][i] for step in tqdm.tqdm(keep_frames + [-1], desc=f"Rendering trajectory {i}", leave=False)]
            M_pred_traj = [m._replace(xyz=m.xyz) for m in M_pred_traj]

            wandb.log({
                f"{folder}/true_{i}": wandb.Html(html_render_molecule(M_trues[i])),
                f"{folder}/pred_{i}": wandb.Html(html_render_molecule(M_preds[i])),
                f"{folder}/anim_pred_{i}": wandb.Html(html_render_trajectory(M_pred_traj)),
                "epoch": self.current_epoch,
            })

        metrics = {f"{folder}/{k}": statistics.mean(vs) for k, vs in metrics.items()}
        self.log_dict(metrics, batch_size=M.batch_size, sync_dist=hp.distributed)
