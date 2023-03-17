import dgl.function as fn
import torch
import torch.linalg as LA
import torch.nn as nn


# Modified from DGL source code:
# https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/egnnconv.py
# Adapted to implement:
# https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/egnn_new.py


class EquivariantBlock(nn.Module):

    def __init__(
        self,
        d_coords, d_hidden, d_edges,
        update_hidden=True,
        equivariance="e3",
    ):
        assert equivariance in {"e3", "reflect"}
        super().__init__()

        self.d_coords = d_coords
        self.d_hidden = d_hidden
        self.d_edges = d_edges
        self.update_hidden = update_hidden

        self.is_e3 = (equivariance == "e3")
        d_dists = 1 if self.is_e3 else (1 + d_coords)
        d_msg_f = (2 * d_hidden) + d_edges + d_dists + (0 if self.is_e3 else 2 * d_coords)

        if update_hidden:
            self.edge_mlp = nn.Sequential(
                nn.Linear(d_msg_f, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.SiLU(),
            )

            self.att_mlp = nn.Sequential(
                nn.Linear(d_hidden, 1),
                nn.Sigmoid(),
            )

            self.node_mlp = nn.Sequential(
                nn.Linear(2 * d_hidden, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, d_hidden),
            )

        self.coord_mlp = nn.Sequential(
            nn.Linear(d_msg_f, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1, bias=False),
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def message(self, edges):
        f = [edges.src["h"], edges.dst["h"], edges.data["radial"], edges.data["a"]]
        if not self.is_e3:
            f = f + [edges.data["x_diff"].abs(), edges.src["x"].abs(), edges.dst["x"].abs()]
        f = torch.cat(f, dim=-1)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(f) * edges.data["x_diff"]

        if self.update_hidden:
            msg_h = self.edge_mlp(f)
            msg["msg_h"] = self.att_mlp(msg_h) * msg_h

        return msg

    def forward(self, G, h, x, a=None):
        with G.local_scope():
            G.ndata["x"] = x  # coordinates
            G.ndata["h"] = h  # hidden feature
            G.edata["a"] = a  # edge features

            # Get coordinate diff & radial features
            G.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            G.edata["radial"] = LA.norm(G.edata["x_diff"], ord=2, dim=-1, keepdim=True)

            # Normalize coordinate difference
            G.edata["x_diff"] = G.edata["x_diff"] / (G.edata["radial"] + 1.0)
            G.apply_edges(self.message)

            G.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_neigh"))
            x_neigh = G.ndata["x_neigh"]
            x = x + x_neigh

            if self.update_hidden:
                G.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))
                h_neigh = G.ndata["h_neigh"]
                h = h + self.node_mlp(torch.cat([h, h_neigh], dim=-1))
            else:
                h = None

        return h, x
