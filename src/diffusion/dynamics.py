import torch
import torch.nn as nn

from src import utils
from src.modules import Activation, EquivariantBlock, LayerNorm


class DummyDynamics(nn.Module):

    def __init__(self):
        super().__init__()

        # Here so there is a parameter to train
        self.eps = nn.Parameter(torch.ones(1))

    def forward(self, M, temb):
        eps = self.eps * torch.randn_like(M.coords)
        return utils.zeroed_com(M, eps, orthogonal=False)


class EquivariantDynamics(nn.Module):

    def __init__(
        self,
        atom_features,
        temb_features,
        cond_features,
        hidden_features,
        num_layers,
        norm_before_blocks,
        norm_hidden_type,
        norm_adaptively,
        zero_com_after_blocks,
        act,
        egnn_distance_fns,
        norm_coords_type,
        **kwargs
    ):
        super().__init__()

        self.egnn_distance_fns = egnn_distance_fns
        self.num_layers = num_layers
        self.norm_before_blocks = norm_before_blocks
        self.norm_adaptively = norm_adaptively
        self.zero_com_after_blocks = zero_com_after_blocks

        # atom emb + time emb + mass + mass_normalized + cond_coords + cond_mask + moments
        self.embed_atom = nn.Embedding(82, atom_features)  # kind of wasteful but makes code simpler
        nf = atom_features + temb_features + 1 + 1 + 3 + 3 + 3
        self.proj_h = nn.Linear(nf, hidden_features)

        if norm_adaptively:
            assert norm_before_blocks
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

        self.egnn = nn.ModuleList()

        for i in range(num_layers):
            blocks = {}

            if norm_before_blocks:
                blocks["norm_hidden"] = LayerNorm(hidden_features, adaptive_features)

            blocks["equivariant"] = EquivariantBlock(
                dim=3,
                hidden_features=hidden_features,
                edge_features=EquivariantBlock.distance_features(3, egnn_distance_fns),
                distance_fns=egnn_distance_fns,
                act=act,
                update_hidden=(i + 1 < num_layers),
            )

            self.egnn.append(nn.ModuleDict(blocks))

    def forward(self, M, temb):
        assert temb.ndim == 2
        utils.assert_zeroed_com(M, M.coords)

        aemb = self.embed_atom(M.atom_nums.squeeze(-1))
        f = [aemb, temb, M.masses / 12.0, M.masses_normalized, M.cond_labels, M.cond_mask.float(), M.moments]
        f = torch.cat(f, dim=-1)

        if self.proj_cond is None:
            cond = None
        else:
            cond = self.proj_cond(f)

        h = self.proj_h(f)
        coords = M.coords

        # Get base edge features
        with torch.no_grad():
            M.graph.ndata["x"] = coords
            M.graph.apply_edges(
                lambda edges: {
                    "a": EquivariantBlock.distances(edges, "x", fns=self.egnn_distance_fns)
                }
            )
            M.graph.ndata.pop("x")
            a = M.graph.edata.pop("a")

        for i, blocks in enumerate(self.egnn):
            if self.norm_before_blocks:
                h = blocks["norm_hidden"](M=M, h=h, y=cond)
            h, coords = blocks["equivariant"](M=M, h=h, coords=coords, a=a)

            if self.zero_com_after_blocks:
                coords = utils.zeroed_com(M, coords, orthogonal=False)

        return coords
