import dgl
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from src.kraitchman import rotated_to_principal_axes


class KraitchmanClassifier(nn.Module):

    def __init__(self, scale, stable):
        super().__init__()

        self.scale = scale
        self.stable = stable

    def forward(self, G):
        G = rotated_to_principal_axes(G, stable=self.stable)

        cond_mask = G.ndata["abs_mask"]
        G.ndata["abs_xyz"] = torch.where(cond_mask.unsqueeze(-1), G.ndata["xyz"].abs(), torch.nan)
        return G

    @torch.enable_grad()
    def grad_log_p_y_given_Gt(self, G_t):
        G_t = G_t.local_var()
        G_t.ndata["xyz"] = G_t.ndata["xyz"].detach().requires_grad_()
        G_rotated = self(G_t)

        sum_logps = 0.0

        for G_true, G_pred in zip(dgl.unbatch(G_t), dgl.unbatch(G_rotated)):
            cond_mask = G_true.ndata["abs_mask"]

            y_true = G_true.ndata["abs_xyz"]
            y_pred = G_pred.ndata["abs_xyz"]

            y_true = y_true[cond_mask]
            y_pred = y_pred[cond_mask]

            logp = Normal(loc=y_pred, scale=self.scale).log_prob(y_true).sum()
            sum_logps = sum_logps + logp

        return torch.autograd.grad((sum_logps,), (G_t.ndata["xyz"],))[0]
