import dgl.function as fn
import dgl.nn.functional
import einops
import torch
import torch.nn as nn

from src.modules.layers import Activation, LayerNorm


def distance_features(dim, equivariance, relaxed):
    assert equivariance in {"e3", "ref"}
    features = 1 + (3 if relaxed else 0)
    if equivariance == "ref":
        features += (dim * (3 if relaxed else 1))
    return features


def distances(edges, feat, equivariance, relaxed, aux=False):
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
            (srcs * dsts).sum(dim=-1, keepdim=True),
        ])
        if equivariance == "ref":
            D.extend([srcs2, dsts2])
    D = torch.cat(D, dim=-1)

    return (D, (diffs, radials)) if aux else D


class EquivariantBlock(nn.Module):

    def __init__(
        self,
        equivariance,
        relaxed,
        dim,
        hidden_features,
        edge_features,
        act,
        update_hidden=True,
    ):
        assert equivariance in {"e3", "ref"}
        super().__init__()

        self.equivariance = equivariance
        self.relaxed = relaxed
        self.update_hidden = update_hidden

        message_features = (2 * hidden_features) + edge_features
        message_features += distance_features(dim, equivariance, relaxed)

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
        D, (diffs, radials) = distances(edges, "x", self.equivariance, self.relaxed, aux=True)
        f = [edges.src["h"], edges.dst["h"], D, edges.data["a"]]
        f = torch.cat(f, dim=-1)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(f) * diffs / (torch.sqrt(radials + 1e-5) + 1.0)

        if self.update_hidden:
            msg_h = self.edge_mlp(f)
            msg["msg_h"] = self.att_mlp(msg_h) * msg_h

        return msg

    def forward(self, M, h, coords, a):
        G = M.graph

        with G.local_scope():
            G.ndata["x"] = coords
            G.ndata["h"] = h
            G.edata["a"] = a
            G.apply_edges(self.message)

            G.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_agg"))
            coords = coords + G.ndata["x_agg"]

            if self.update_hidden:
                G.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_agg"))
                h_agg = G.ndata["h_agg"]
                h = h + self.node_mlp(torch.cat([h, h_agg], dim=-1))
            else:
                h = None

        return h, coords


class EquivariantTransformerBlock(nn.Module):

    def __init__(
        self,
        equivariance,
        relaxed,
        dim,
        hidden_features,
        edge_features,
        adaptive_features,
        num_heads,
        act,
        update_hidden=True,
    ):
        assert equivariance in {"e3", "ref"}
        assert hidden_features % num_heads == 0
        super().__init__()

        self.equivariance = equivariance
        self.relaxed = relaxed
        self.num_heads = num_heads
        self.update_hidden = update_hidden

        inner_features = int(1.25 * hidden_features)
        message_features = (2 * hidden_features) + edge_features
        message_features += distance_features(dim, equivariance, relaxed)

        self.edge_mlp = nn.Sequential(
            nn.Linear(message_features, inner_features),
            Activation(act),
            nn.Linear(inner_features, inner_features),
            Activation(act),
        )

        scale_dim = 1 if (equivariance == "e3") else dim
        self.coord_mlp = nn.Linear(inner_features, scale_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.coord_mlp.weight, gain=0.001)

        if update_hidden:
            self.val_mlp = nn.Linear(inner_features, hidden_features, bias=False)
            self.att_mlp = nn.Linear(inner_features, num_heads, bias=False)
            self.out_mlp = nn.Linear(hidden_features, hidden_features, bias=False)

            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_features, inner_features),
                Activation(act),
                nn.Linear(inner_features, hidden_features),
            )

        self.norm_agg = LayerNorm(hidden_features, adaptive_features)
        self.norm_ff = LayerNorm(hidden_features, adaptive_features)

    def message(self, edges):
        D, (diffs, radials) = distances(edges, "x", self.equivariance, self.relaxed, aux=True)
        f = [edges.src["h"], edges.dst["h"], D, edges.data["a"]]
        f = torch.cat(f, dim=-1)
        m = self.edge_mlp(f)

        msg = dict()
        msg["msg_x"] = self.coord_mlp(m) * diffs / (torch.sqrt(radials + 1e-5) + 1.0)

        if self.update_hidden:
            msg["msg_attn"] = einops.rearrange(self.att_mlp(m), "e h -> e h 1")
            msg["msg_h"] = einops.rearrange(self.val_mlp(m), "e (h c) -> e h c", h=self.num_heads)

        return msg

    def forward(self, M, h, coords, a, y=None, res=None):
        G = M.graph

        with G.local_scope():
            G.ndata["x"] = coords
            G.ndata["h"] = h
            G.edata["a"] = a
            G.apply_edges(self.message)

            G.update_all(fn.copy_e("msg_x", "m"), fn.sum("m", "x_agg"))
            coords = coords + G.ndata["x_agg"]

            if self.update_hidden:
                weights = dgl.nn.functional.edge_softmax(G, G.edata["msg_attn"])
                G.edata["msg_h"] = weights * G.edata["msg_h"]
                G.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_agg"))
                h_agg = einops.rearrange(G.ndata["h_agg"], "n h c -> n (h c)")

                f = self.out_mlp(h_agg)
                res = res + f
                h = self.norm_agg(h + f, y=y)

                f = self.node_mlp(h)
                res = res + f
                h = self.norm_ff(h + f, y=y)
            else:
                h = None

        return h, coords, res
