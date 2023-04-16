import dgl
import torch
import torch.nn as nn
from torch.nn import init


class LayerNorm(nn.Module):

    def __init__(self, hidden_features, adaptive_features):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_features)
        self.proj_ada = nn.Linear(adaptive_features, 2 * hidden_features) if (adaptive_features > 0) else None

    def forward(self, M, h, y):
        h = self.norm(h)
        if self.proj_ada is not None:
            params = self.proj_ada(y)
            scale, shift = params.chunk(chunks=2, dim=-1)
            h = torch.addcmul(shift, h, scale + 1)
        return h


# from https://github.com/cvignac/MiDi/blob/145ca8bc0d5962e6ef52025fe8d4b9f0195ecd6b/src/models/layers.py
class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """ Note: There is a relatively similar layer implemented by NVIDIA:
            https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
            It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.normalized_shape = (1,)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, G, feat):  # feat is usually xyz
        with G.local_scope():
            G.ndata['norm'] = torch.norm(G.ndata[feat], dim=-1, keepdim=True)  # (N 1)
            mean_norm = dgl.broadcast_nodes(G, dgl.mean_nodes(G, 'norm'))  # (N 1)
            new_pos = self.weight * G.ndata[feat] / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
