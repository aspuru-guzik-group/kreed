import dgl.function as fn
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers import Activation


class EquivariantBlock(nn.Module):

    DISTANCE_FN_REGISTRY = {"dist_l2", "dist_l1", "dist_abs", "dist_cos", "orig_l2", "orig_l1", "orig_abs"}

    @classmethod
    def distance_features(cls, dim, fns):
        assert set(fns) <= cls.DISTANCE_FN_REGISTRY
        increments = {
            "dist_l2": 1, "dist_l1": 1, "dist_abs": dim, "dist_cos": 1,
            "orig_l2": 2, "orig_l1": 2, "orig_abs": 2 * dim,
        }
        return sum(i for k, i in increments.items() if (k in fns))

    @classmethod
    def distances(cls, edges, feat, fns, aux=False):
        srcs = edges.src[feat]
        dsts = edges.dst[feat]
        diffs = srcs - dsts
        radials = LA.norm(diffs, ord=2, dim=-1, keepdim=True)

        D = []
        if "dist_l2" in fns:
            D.append(radials)
        if "dist_l1" in fns:
            D.append(LA.norm(diffs, ord=1, dim=-1, keepdim=True))
        if "dist_abs" in fns:
            D.append(diffs.abs())
        if "dist_cos" in fns:
            D.append(F.cosine_similarity(srcs, dsts, dim=-1).unsqueeze(-1))
        if "orig_l2" in fns:
            D.append(LA.norm(srcs, ord=2, dim=-1, keepdim=True))
            D.append(LA.norm(dsts, ord=2, dim=-1, keepdim=True))
        if "orig_l1" in fns:
            D.append(LA.norm(srcs, ord=1, dim=-1, keepdim=True))
            D.append(LA.norm(dsts, ord=1, dim=-1, keepdim=True))
        if "orig_abs" in fns:
            D.append(srcs.abs())
            D.append(dsts.abs())
        D = torch.cat(D, dim=-1)

        return (D, (diffs, radials)) if aux else D

    def __init__(
        self,
        dim,
        hidden_features,
        edge_features,
        distance_fns,
        act,
        update_hidden=True,
    ):
        super().__init__()

        self.distance_fns = distance_fns
        self.update_hidden = update_hidden

        message_features = (2 * hidden_features) + edge_features + self.distance_features(dim, distance_fns)

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

        self.coord_mlp = nn.Sequential(
            nn.Linear(message_features, hidden_features),
            Activation(act),
            nn.Linear(hidden_features, hidden_features),
            Activation(act),
            nn.Linear(hidden_features, 1, bias=False),  # TODO: try dim instead of 1?
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def message(self, edges):
        D, (diffs, radials) = self.distances(edges, "x", fns=self.distance_fns, aux=True)
        f = [edges.src["h"], edges.dst["h"], D, edges.data["a"]]
        f = torch.cat(f, dim=-1)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(f) * diffs / (radials + 1.0)

        if self.update_hidden:
            msg_h = self.edge_mlp(f)
            msg["msg_h"] = self.att_mlp(msg_h) * msg_h

        return msg

    def forward(self, M, h, coords, a=None):
        G = M.graph

        with G.local_scope():
            G.ndata["x"] = coords  # coordinates
            G.ndata["h"] = h  # hidden feature
            G.edata["a"] = a  # edge features
            G.apply_edges(self.message)

            G.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_agg"))
            coords = coords + G.ndata["x_agg"]

            if self.update_hidden:
                G.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_agg"))
                h_agg = torch.cat([h, G.ndata["h_agg"]], dim=-1)
                h = h + self.node_mlp(h_agg)
            else:
                h = None

        return h, coords
