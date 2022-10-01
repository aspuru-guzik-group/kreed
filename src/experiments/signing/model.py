import dgl
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SetTransformerEncoder
from torch import nn


class CoordinateSignPredictor(nn.Module):

    def __init__(self, d_embed, d_model, n_heads, n_layers):
        super().__init__()

        assert (d_model % n_heads) == 0

        self.embed_atom_num = nn.Embedding(50, d_embed)  # TODO: setting to 50 is wasteful

        self.fc_proj = nn.Sequential(
            nn.Linear(d_embed + 3 + 1, d_model),
            nn.ReLU(),
        )

        self.encoder = SetTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_head=(d_model // n_heads),
            d_ff=(2 * d_model),
            n_layers=n_layers,
        )

        self.fc_out = nn.Linear(d_model, 3)

    def forward(self, G):
        feats = torch.cat(
            [
                self.embed_atom_num(G.ndata["atom_nums"]),
                G.ndata["coords"],
                G.ndata["mask"].unsqueeze(-1).float(),
            ],
            dim=-1
        )

        x = self.fc_proj(feats)
        z = self.encoder(G, x)
        logits = self.fc_out(z)
        return logits


class PLCoordinateSignPredictor(pl.LightningModule):

    def __init__(
        self,
        d_embed, d_model, n_heads, n_layers,
        lr,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.predictor = CoordinateSignPredictor(
            d_embed=d_embed,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _step(self, G, split):
        logits = self.predictor(G)
        preds = (logits > 0.0).detach().float()

        metric_kwargs = {
            "G": G,
            "orig_labels": G.ndata["labels"],
            "flip_labels": (1.0 - G.ndata["labels"]),
            "mask": G.ndata["mask"],
            "num_unmasked": torch.sum(~G.ndata["mask"]).int()
        }

        loss = self._compute_and_reduce_metric(
            metric_fn=lambda y, t: F.binary_cross_entropy_with_logits(y, t, reduction="none"),
            preds=logits,
            **metric_kwargs,
        )

        with torch.no_grad():
            acc = self._compute_and_reduce_metric(
                metric_fn=lambda y, t: (y == t).float(),
                preds=preds,
                mode="max",
                **metric_kwargs,
            )

        batch_size = metric_kwargs["num_unmasked"]
        self.log(f"{split}_loss", loss, batch_size=batch_size)
        self.log(f"{split}_acc", acc, batch_size=batch_size)

        return loss

    def _compute_and_reduce_metric(self, metric_fn, G, preds, orig_labels, flip_labels, mask, num_unmasked, mode="min"):
        orig_metric = metric_fn(preds, orig_labels)
        flip_metric = metric_fn(preds, flip_labels)

        orig_metric[mask, :] = 0.0
        flip_metric[mask, :] = 0.0

        agg_metric = 0.0
        agg_fn = torch.minimum if (mode == "min") else torch.maximum

        for i, ax in enumerate(["x, y, z"]):
            G.ndata[f"orig_{ax}"] = orig_metric[:, i]
            G.ndata[f"flip_{ax}"] = flip_metric[:, i]

            agg_metric += agg_fn(
                dgl.sum_nodes(G, f"orig_{ax}"),
                dgl.sum_nodes(G, f"flip_{ax}")
            ).sum()

            # cleanup
            G.ndata.pop(f"orig_{ax}")
            G.ndata.pop(f"flip_{ax}")

        return agg_metric / (3 * num_unmasked)
