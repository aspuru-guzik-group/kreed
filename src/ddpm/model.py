from torch import nn
from dgl.nn.pytorch.conv import EGNNConv


class EGNNDynamics(nn.Module):

    def __int__(
        self,
        d_embed_atom,
        d_hidden,
        n_layers,
    ):

        self.lin_hid = nn.Linear(3 + d_embed_atom, d_hidden)

        self.egnn_layers = nn.ModuleList([
            EGNNConv(in_size=d_hidden, hidden_size=d_hidden, out_size=d_hidden, edge_feat_size=2)
            for _ in range(n_layers)
        ])
