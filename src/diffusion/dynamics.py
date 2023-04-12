import dgl
import dgl.function as fn
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from src.modules import EquivariantBlock


class EquivariantDynamics(nn.Module):

    def __init__(self, equivariance, d_atom_vocab, d_hidden, n_layers):
        assert equivariance in {"e3", "reflect"}
        super().__init__()

        self.d_atom_vocab = d_atom_vocab
        self.proj_h = nn.Linear(11 + d_atom_vocab, d_hidden)
        self.is_e3 = (equivariance == "e3")

        self.eq_blocks = nn.ModuleList([
            EquivariantBlock(
                d_coords=3,
                d_hidden=d_hidden,
                d_edges=(1 if self.is_e3 else 4),
                update_hidden=(i + 1 < n_layers),
                equivariance=equivariance,
            )
            for i in range(n_layers)
        ])

    def featurize_nodes(self, G, t):
        assert (t.ndim == 1) and torch.is_floating_point(t)
        xyz = G.ndata["xyz"]  # for casting

        # Node features
        atom_one_hots = F.one_hot(G.ndata["atom_ids"].squeeze(-1), num_classes=self.d_atom_vocab).to(xyz)  # (N d_vocab)
        atom_masses = G.ndata["atom_masses"].to(xyz)  # (N 1)
        temb = dgl.broadcast_nodes(G, t).to(xyz)  # (N)

        # Conditioning features
        cond_labels = G.ndata["cond_labels"].to(xyz)  # (N 3)
        cond_mask = G.ndata["cond_mask"].to(xyz)  # (N 3)

        # moments
        moments = G.ndata["moments"].to(xyz)  # (N 3)
        moments_per_node = (moments / dgl.broadcast_nodes(G, G.batch_num_nodes()).unsqueeze(-1))

        features = [
            atom_one_hots,
            atom_masses / 12.0,  # FIXME: the normalization is arbitrary here
            temb.unsqueeze(-1),
            cond_mask,
            cond_labels,
            moments_per_node / 12.0,
        ]

        return torch.cat(features, dim=-1)  # (N d_vocab+11)

    def forward(self, G, t):
        xyz = G.ndata["xyz"] # (N 3)

        with G.local_scope():
            G.apply_edges(fn.u_sub_v("xyz", "xyz", "xyz_diff"))
            a = [LA.norm(G.edata["xyz_diff"], ord=2, dim=-1, keepdim=True)]
            if not self.is_e3:
                a = a + [G.edata["xyz_diff"].abs()]
            a = torch.cat(a, dim=-1)

        h = self.featurize_nodes(G=G, t=t)
        h = self.proj_h(h)
        for block in self.eq_blocks:
            h, xyz = block(G, h=h, x=xyz, a=a)
        vel = xyz - G.ndata["xyz"]

        return utils.zeroed_weighted_com(G, vel)
