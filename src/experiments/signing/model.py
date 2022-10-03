import dgl
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SetTransformerEncoder
from torch import nn


class CoordinateSignPredictor(nn.Module):

    def __init__(self, d_embed, d_vocab, d_model, n_heads, n_layers):
        super().__init__()

        assert (d_model % n_heads) == 0

        self.d_embed = d_embed
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embed_atom_num = nn.Embedding(118, d_embed)  # TODO: setting to 118 is wasteful

        self.fc_proj = nn.Sequential(
            nn.Linear(d_embed + d_vocab + 3, d_model),
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

    def forward(self, G, formula):
        mol_comp = formula / G.batch_num_nodes().unsqueeze(-1)  # normalize
        mol_comp = torch.repeat_interleave(mol_comp, repeats=G.batch_num_nodes(), dim=0)  # broadcast to ndata shape

        feats = torch.cat(
            [
                self.embed_atom_num(G.ndata["atom_nums"]),
                mol_comp,
                G.ndata["coords"],
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
        d_embed, d_vocab, d_model, n_heads, n_layers,
        lr,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.predictor = CoordinateSignPredictor(
            d_embed=d_embed,
            d_vocab=d_vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        return self._step(*batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self._step(*batch, split="val")

    def test_step(self, batch, batch_idx):
        return self._step(*batch, split="test")

    def _step(self, G, formula, split):
        logits = self.predictor(G, formula)
        preds = (logits > 0.0).detach().float()

        metric_kwargs = {
            "G": G,
            "orig_labels": G.ndata["labels"],
            "flip_labels": (1.0 - G.ndata["labels"]),
        }

        loss, _ = self._compute_and_reduce_metric(
            metric_fn=lambda y, t: F.binary_cross_entropy_with_logits(y, t, reduction="none"),
            preds=logits,
            **metric_kwargs,
        )

        with torch.no_grad():
            acc, agg_accs = self._compute_and_reduce_metric(
                metric_fn=lambda y, t: (y == t).float(),
                preds=preds,
                take_max=True,
                **metric_kwargs,
            )

            mol_acc = (agg_accs == (3 * G.batch_num_nodes())).float().mean()

        self.log(f"{split}_loss", loss, batch_size=logits.shape[0])
        self.log(f"{split}_acc", acc, batch_size=logits.shape[0])
        self.log(f"{split}_mol_acc", mol_acc, batch_size=G.batch_size)

        return loss

    def _compute_and_reduce_metric(self, metric_fn, G, preds, orig_labels, flip_labels, take_max=False):
        G.ndata["orig"] = metric_fn(preds, orig_labels)
        G.ndata["flip"] = metric_fn(preds, flip_labels)

        agg_fn = torch.maximum if take_max else torch.minimum

        agg_metrics = agg_fn(
            dgl.sum_nodes(G, "orig"),
            dgl.sum_nodes(G, "flip"),
        ).sum(dim=-1)

        # cleanup
        G.ndata.pop(f"orig")
        G.ndata.pop(f"flip")

        avg_metric = torch.sum(agg_metrics) / preds.numel()
        return avg_metric, agg_metrics
