import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from src.kraitchman import ATOM_MASSES


# Modified from DGL source code:
# https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/egnnconv.py
# Adapted to implement:
# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/egnn_new.py


class EGNNConv(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, equivariance='e3'):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        self.equivariance = equivariance

        if self.equivariance == 'e3':
            self.norm = lambda x: x.square()
        elif self.equivariance == 'reflection':
            self.norm = lambda x: x.abs()

        act_fn = nn.SiLU()

        if out_size > 0:  # no hidden features outputted if out_size = 0

            # \phi_e
            self.edge_mlp = nn.Sequential(
                # +1 for the radial feature: ||x_i - x_j||^2
                nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
                act_fn,
                nn.Linear(hidden_size, hidden_size),
                act_fn,
            )

            # \phi_inf
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

            # \phi_h
            self.node_mlp = nn.Sequential(
                nn.Linear(in_size + hidden_size, hidden_size),
                act_fn,
                nn.Linear(hidden_size, out_size),
            )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def message(self, edges):
        f = [edges.src["h"], edges.dst["h"], edges.data["radial"]]
        if self.edge_feat_size > 0:
            f = f + [edges.data["a"]]
        f = torch.cat(f, dim=-1)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(f) * edges.data["x_diff"]

        if self.out_size > 0:
            msg_h = self.edge_mlp(f)
            msg["msg_h"] = self.att_mlp(msg_h) * msg_h

        return msg

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        with graph.local_scope():
            # node feature
            graph.ndata["h"] = node_feat

            # coordinate feature
            graph.ndata["x"] = coord_feat

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat

            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["radial"] = self.norm(graph.edata["x_diff"]).sum(dim=1).unsqueeze(-1)

            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (graph.edata["radial"].sqrt() + 1.0)
            graph.apply_edges(self.message)

            graph.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"))
            x_neigh = graph.ndata["x_neigh"]
            x = coord_feat + x_neigh

            if self.out_size > 0:
                graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))
                h_neigh = graph.ndata["h_neigh"]
                h = node_feat + self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            else:
                h = None

            return h, x


class EGNNDynamics(nn.Module):

    def __init__(
        self,
        d_atom_vocab,
        d_hidden,
        n_layers,
        equivariance,
    ):
        super().__init__()

        self.d_atom_vocab = d_atom_vocab
        self.lin_hid = nn.Linear(6 + d_atom_vocab, d_hidden)
        self.equivariance = equivariance

        if self.equivariance == 'e3':
            self.norm = lambda x: x.square()
            self.maybe_center = lambda xyz, G: utils.centered_mean(G, xyz - G.ndata["xyz"])
        elif self.equivariance == 'reflection':
            self.norm = lambda x: x.abs()
            self.maybe_center = lambda xyz, G: xyz

        self.egnn_layers = nn.ModuleList([
            EGNNConv(
                in_size=d_hidden,
                hidden_size=d_hidden,
                out_size=(0 if (i + 1 == n_layers) else d_hidden),
                edge_feat_size=1,
                equivariance=self.equivariance,
            )
            for i in range(n_layers)
        ])

    def _featurize_nodes(self, G, t):
        xyz = G.ndata["xyz"]  # for casting

        # Node features
        atom_ids = F.one_hot(G.ndata["atom_ids"], num_classes=self.d_atom_vocab).to(xyz)  # (N d_vocab)
        atom_masses = ATOM_MASSES[G.ndata["atom_nums"].cpu()].to(xyz)  # (N)
        temb = dgl.broadcast_nodes(G, t).to(xyz)  # (N 1)

        # Conditioning features
        cond_mask = G.ndata["abs_node_mask"].to(xyz)  # (N)
        abs_xyz = G.ndata["abs_xyz"].to(xyz)  # (N 3)

        features = [
            atom_ids,
            atom_masses.unsqueeze(-1) / 12.0,  # FIXME: the normalization is arbitrary here
            temb,
            cond_mask.unsqueeze(-1),
            abs_xyz,
        ]

        return torch.cat(features, dim=-1)  # (N d_vocab+6)

    def forward(self, G, t):
        xyz = G.ndata["xyz"]
        h = self._featurize_nodes(G=G, t=t)

        with G.local_scope():
            G.apply_edges(fn.u_sub_v("xyz", "xyz", "xyz_diff"))
            a = self.norm(G.edata["xyz_diff"]).sum(dim=1).unsqueeze(-1)

        h = self.lin_hid(h)
        for layer in self.egnn_layers:
            h, xyz = layer(G, node_feat=h, coord_feat=xyz, edge_feat=a)
        return self.maybe_center(xyz, G)
