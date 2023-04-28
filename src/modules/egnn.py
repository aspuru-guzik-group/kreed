import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.ops import Activation
from src.modules.normalizations import GraphNorm, LayerNorm, SE3Norm


class EquivariantBlock(nn.Module):

    @classmethod
    def distance_features(cls, dim, equivariance, relaxed):
        assert equivariance in {"e3", "ref"}
        features = 1 + (3 if relaxed else 0)
        if equivariance == "ref":
            features += (dim * (3 if relaxed else 1))
        return features

    @classmethod
    def distances(cls, edges, feat, equivariance, relaxed, aux=False):
        assert equivariance in {"e3", "ref"}
        srcs = edges.src[feat]
        dsts = edges.dst[feat]
        diffs = srcs - dsts

        diffs2 = diffs.square()
        radials = diffs2.sum(dim=-1, keepdim=True)

        D = [radials]
        if equivariance == "ref":
            D.append(diffs2)
        if relaxed:
            srcs2 = srcs.square()
            dsts2 = dsts.square()
            D.extend([
                srcs2.sum(dim=-1, keepdim=True),
                dsts2.sum(dim=-1, keepdim=True),
                F.cosine_similarity(srcs, dsts, dim=-1).unsqueeze(-1),
            ])
            if equivariance == "ref":
                D.extend([srcs2, dsts2])
        D = torch.cat(D, dim=-1)

        return (D, (diffs, radials)) if aux else D

    def __init__(
        self,
        equivariance,
        relaxed,
        dim,
        hidden_features,
        edge_features,
        adaptive_features,
        norm_coords,
        norm_hidden,
        act,
        update_hidden=True,
    ):
        assert equivariance in {"e3", "ref"}
        super().__init__()

        self.equivariance = equivariance
        self.relaxed = relaxed
        self.update_hidden = update_hidden

        message_features = (2 * hidden_features) + edge_features
        message_features += self.distance_features(dim, equivariance, relaxed)

        if norm_coords == "se3":
            self.norm_coords = SE3Norm(adaptive_features)
        else:
            self.norm_coords = None

        if norm_hidden == "none":
            self.norm_h = self.norm_h_agg = None
        else:
            if norm_hidden == "layer":
                Norm = LayerNorm
            elif norm_hidden == "graph":
                Norm = GraphNorm
            else:
                raise ValueError()
            self.norm_h = Norm(hidden_features, adaptive_features)
            self.norm_h_agg = Norm(hidden_features, adaptive_features) if update_hidden else None

        if update_hidden:
            self.edge_mlp = nn.Sequential(
                nn.Linear(message_features, hidden_features),
                Activation(act),
                nn.Linear(hidden_features, hidden_features),
                Activation(act),
            )

            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_features, 1),
                nn.Sigmoid(),
            )

            self.node_mlp = nn.Sequential(
                nn.Linear(2 * hidden_features, hidden_features),
                Activation(act),
                nn.Linear(hidden_features, hidden_features),
            )

        scale_dim = 1 if (equivariance == "e3") else dim
        self.coord_mlp = nn.Sequential(
            nn.Linear(message_features, hidden_features),
            Activation(act),
            nn.Linear(hidden_features, hidden_features),
            Activation(act),
            nn.Linear(hidden_features, scale_dim, bias=False),
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def message(self, edges):
        D, (diffs, radials) = self.distances(edges, "x", self.equivariance, self.relaxed, aux=True)
        f = [edges.src["h"], edges.dst["h"], D, edges.data["a"]]
        f = torch.cat(f, dim=-1)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(f) * diffs / (radials + 1.0)

        if self.update_hidden:
            msg_h = self.edge_mlp(f)
            msg["msg_h"] = self.att_mlp(msg_h) * msg_h

        return msg

    def forward(self, M, h, coords, a=None, y=None):
        G = M.graph

        with G.local_scope():
            G.ndata["x"] = coords
            G.ndata["h"] = h if (self.norm_h is None) else self.norm_h(M, h=h, y=y)
            G.edata["a"] = a
            G.apply_edges(self.message)

            G.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_agg"))
            x_agg = G.ndata["x_agg"]
            x_agg = x_agg if (self.norm_coords is None) else self.norm_coords(M, coords=x_agg, y=y)
            coords = coords + x_agg

            if self.update_hidden:
                G.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_agg"))
                h_agg = G.ndata["h_agg"]
                h_agg = h_agg if (self.norm_h_agg is None) else self.norm_h_agg(M, h=h_agg, y=y)
                h = h + self.node_mlp(torch.cat([G.ndata["h"], h_agg], dim=-1))
            else:
                h = None

        return h, coords
