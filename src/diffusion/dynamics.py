import dgl
import torch
import torch.nn as nn

from src import utils
from src.modules import (
    Activation,
    EquivariantBlock,
    EquivariantTransformerBlock,
    LayerNorm,
    distance_features,
    distances,
)


class DummyDynamics(nn.Module):

    def __init__(self):
        super().__init__()

        # Here so there is a parameter to train
        self.eps = nn.Parameter(torch.ones(1))

    def forward(self, M, temb):
        eps = self.eps * torch.randn_like(M.coords)
        return utils.zeroed_com(M, eps, orthogonal=False)


class EDMDynamics(nn.Module):

    def __init__(
        self,
        atom_features,
        hidden_features,
        num_layers,
        act,
        egnn_equivariance,
        egnn_relaxed,
        zero_com_before_blocks,
        **kwargs
    ):
        super().__init__()

        self.equivariance = egnn_equivariance
        self.relaxed = egnn_relaxed
        self.num_layers = num_layers
        self.zero_com_before_blocks = zero_com_before_blocks

        # atom emb + time emb + mass + mass_normalized + cond_coords + cond_mask + moments
        self.embed_atom = nn.Embedding(90, atom_features)  # kind of wasteful but makes code simpler
        nf = atom_features + 1 + 1 + 1 + 3 + 3 + 3
        self.proj_h = nn.Linear(nf, hidden_features)

        self.egnn = nn.ModuleList([
            EquivariantBlock(
                equivariance=egnn_equivariance,
                relaxed=egnn_relaxed,
                dim=3,
                hidden_features=hidden_features,
                edge_features=distance_features(3, egnn_equivariance, egnn_relaxed),
                act=act,
                update_hidden=(i + 1 < num_layers),
            )
            for i in range(num_layers)
        ])

    def forward(self, M, temb):
        assert (temb.ndim == 2) and (temb.shape[1] == 1)
        utils.assert_zeroed_com(M, M.coords)

        aemb = self.embed_atom(M.atom_nums.squeeze(-1))
        f = [aemb, temb, M.masses, M.masses_normalized, M.cond_labels, M.cond_mask.float(), M.moments]
        f = torch.cat(f, dim=-1)

        h = self.proj_h(f)
        coords = M.coords

        # Get base edge features
        with torch.no_grad():
            M.graph.ndata["x"] = coords
            M.graph.apply_edges(
                lambda edges: {
                    "a": distances(edges, "x", self.equivariance, self.relaxed)
                }
            )
            M.graph.ndata.pop("x")
            a = M.graph.edata.pop("a")

        for i, block in enumerate(self.egnn):
            if self.zero_com_before_blocks:
                coords = utils.zeroed_com(M, coords, orthogonal=False)
            h, coords = block(M=M, h=h, coords=coords, a=a)

        return utils.zeroed_com(M, coords, orthogonal=False)


class TransformerDynamics(nn.Module):

    def __init__(
        self,
        atom_features,
        temb_features,
        cond_features,
        hidden_features,
        inner_features,
        num_layers,
        num_attn_heads,
        norm_adaptively,
        act,
        egnn_equivariance,
        egnn_relaxed,
        zero_com_before_blocks,
        **kwargs
    ):
        super().__init__()

        self.equivariance = egnn_equivariance
        self.relaxed = egnn_relaxed
        self.num_layers = num_layers
        self.norm_adaptively = norm_adaptively
        self.zero_com_before_blocks = zero_com_before_blocks

        # atom emb + time emb + mass + mass_normalized + cond_coords + cond_mask + moments
        self.embed_atom = nn.Embedding(90, atom_features)  # kind of wasteful but makes code simpler
        nf = atom_features + temb_features + 1 + 1 + 3 + 3 + 3
        self.proj_h = nn.Linear(nf, hidden_features)

        if norm_adaptively:
            adaptive_features = cond_features
            self.proj_cond = nn.Sequential(
                nn.Linear(nf, cond_features),
                Activation(act),
                nn.Linear(cond_features, cond_features),
                Activation(act),
            )
        else:
            adaptive_features = -1
            self.proj_cond = None

        self.egnn = nn.ModuleList([
            EquivariantTransformerBlock(
                equivariance=egnn_equivariance,
                relaxed=egnn_relaxed,
                dim=3,
                hidden_features=hidden_features,
                inner_features=inner_features,
                edge_features=distance_features(3, egnn_equivariance, egnn_relaxed),
                adaptive_features=adaptive_features,
                num_heads=num_attn_heads,
                act=act,
                update_hidden=(i + 1 < num_layers),
            )
            for i in range(num_layers)
        ])

        self.norm = LayerNorm(hidden_features, adaptive_features)

    def forward(self, M, temb):
        assert temb.ndim == 2
        utils.assert_zeroed_com(M, M.coords)

        aemb = self.embed_atom(M.atom_nums.squeeze(-1))
        f = [aemb, temb, M.masses, M.masses_normalized, M.cond_labels, M.cond_mask.float(), M.moments]
        f = torch.cat(f, dim=-1)

        h = self.proj_h(f)
        coords = M.coords
        cond = None if (self.proj_cond is None) else self.proj_cond(f)

        # Get base edge features
        with torch.no_grad():
            M.graph.ndata["x"] = coords
            M.graph.apply_edges(
                lambda edges: {
                    "a": distances(edges, "x", self.equivariance, self.relaxed)
                }
            )
            M.graph.ndata.pop("x")
            a = M.graph.edata.pop("a")

        res = h

        for i, block in enumerate(self.egnn):
            if self.zero_com_before_blocks:
                coords = utils.zeroed_com(M, coords, orthogonal=False)
            h, coords, res = block(M=M, h=h, coords=coords, a=a, y=cond, res=res)

            if i == self.num_layers - 2:
                h = h + self.norm(res, y=cond)
                res = None

        return utils.zeroed_com(M, coords, orthogonal=False)
