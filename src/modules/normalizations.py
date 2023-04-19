import torch
import torch.nn as nn


class GraphNorm(nn.Module):

    def __init__(self, hidden_features, adaptive_features):
        super().__init__()

        self.adaptive = adaptive_features > 0
        self.alpha = nn.Parameter(torch.ones(hidden_features))
        if self.adaptive:
            self.proj_ada = nn.Linear(adaptive_features, 2 * hidden_features) if self.adaptive else None
            self.weight = self.bias = None
        else:
            self.proj_ada = None
            self.weight = nn.Parameter(torch.zeros(hidden_features))
            self.bias = nn.Parameter(torch.zeros(hidden_features))

    def forward(self, M, h, y):
        mu = M.mean_pool(h, broadcast=True)
        h = h - self.alpha * mu
        var = M.mean_pool(h.square(), broadcast=True)
        h = h / (var + 1e-5).sqrt()

        # Affine
        if self.adaptive:
            params = self.proj_ada(y)
            scale, shift = params.chunk(chunks=2, dim=-1)
        else:
            scale, shift = self.weight, self.bias
        return torch.addcmul(shift, h, scale + 1)


class LayerNorm(nn.Module):

    def __init__(self, hidden_features, adaptive_features):
        super().__init__()

        self.adaptive = adaptive_features > 0
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=(not self.adaptive))
        self.proj_ada = nn.Linear(adaptive_features, 2 * hidden_features) if self.adaptive else None

    def forward(self, M, h, y):
        h = self.norm(h)
        if self.adaptive:
            params = self.proj_ada(y)
            scale, shift = params.chunk(chunks=2, dim=-1)
            h = torch.addcmul(shift, h, scale + 1)
        return h
