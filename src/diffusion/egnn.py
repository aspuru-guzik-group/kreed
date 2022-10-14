import dgl
import dgl.function as fn
import torch
import torch.nn as nn


# Modified from DGL source code:
# https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/egnnconv.py
# Adapted to implement:
# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/egnn_new.py


class EGNNConv(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

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

        msg_h = self.edge_mlp(f)
        msg_h = self.att_mlp(msg_h) * msg_h

        msg_x = self.coord_mlp(f) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

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
            graph.edata["radial"] = (graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1))

            # normalize coordinate difference
            graph.edata["x_diff"] = graph.edata["x_diff"] / (graph.edata["radial"].sqrt() + 1.0)
            graph.apply_edges(self.message)

            graph.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            h = node_feat + self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            x = coord_feat + x_neigh

            return h, x


class EGNNDynamics(nn.Module):

    def __init__(
        self,
        d_embed_atom,
        d_hidden,
        n_layers,
    ):
        super().__init__()

        self.embed_atom = nn.Embedding(100, d_embed_atom)
        self.lin_hid = nn.Linear(2 + 3 + d_embed_atom, d_hidden)

        self.egnn_layers = nn.ModuleList([
            EGNNConv(in_size=d_hidden, hidden_size=d_hidden, out_size=d_hidden, edge_feat_size=1)
            for _ in range(n_layers)
        ])

    def forward(self, G, t):
        xyz = G.ndata["xyz"]
        h = torch.cat(
            [
                self.embed_atom(G.ndata["atom_nums"]),
                dgl.broadcast_nodes(G, t).unsqueeze(-1).float(),
                G.ndata["abs_mask"].unsqueeze(-1).float(),
                G.ndata["abs_xyz"],
            ],
            dim=-1
        )

        with G.local_scope():
            G.apply_edges(fn.u_sub_v("xyz", "xyz", "xyz_diff"))
            a = G.edata["xyz_diff"].square().sum(dim=1).unsqueeze(-1)

        h = self.lin_hid(h)
        for layer in self.egnn_layers:
            h, xyz = layer(G, node_feat=h, coord_feat=xyz, edge_feat=a)

        with G.local_scope():
            G.ndata["vel"] = xyz - G.ndata["xyz"]
            com = dgl.mean_nodes(G, "vel")
            vel = G.ndata["vel"] - dgl.broadcast_nodes(G, com)

        return vel