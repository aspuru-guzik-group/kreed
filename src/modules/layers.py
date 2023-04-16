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

