import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):

    ACTIVATION_REGISTRY = {
        "silu": F.silu,
        "gelu": F.gelu,
        "gelu_approx": functools.partial(F.gelu, approximate="tanh"),
        "relu": F.relu,
        "none": lambda z: z,
    }

    def __init__(self, name):
        super().__init__()

        self.fn = self.ACTIVATION_REGISTRY[name]

    def forward(self, x):
        return self.fn(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        half_dim = embedding_dim // 2
        scale = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(scale * torch.arange(half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        temb = t.float() * self.freqs
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        return temb


class LayerNorm(nn.Module):

    def __init__(self, hidden_features, adaptive_features=-1):
        super().__init__()

        self.adaptive = adaptive_features > 0
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=(not self.adaptive))
        self.proj_ada = nn.Linear(adaptive_features, 2 * hidden_features) if self.adaptive else None

    def forward(self, h, y):
        h = self.norm(h)
        if self.adaptive:
            params = self.proj_ada(y)
            scale, shift = params.chunk(chunks=2, dim=-1)
            h = torch.addcmul(shift, h, scale + 1)
        return h
