import abc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_noise_schedule(alphas_cumprod, clip_margin=0.001):
    """For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.
    """

    alphas_cumprod = np.concatenate([np.ones(1), alphas_cumprod], axis=0)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = np.clip(alphas, a_min=clip_margin, a_max=(1 - clip_margin))
    return np.cumprod(alphas, axis=0)


def polynomial_schedule(T, s=1e-5, power=2.0):
    """A noise schedule based on a simple polynomial equation: 1 - x^power.
    """

    t = np.linspace(1, T, T)
    f = (1 - np.power(t / T, power)) ** 2
    f = clip_noise_schedule(f, clip_margin=0.001)
    alphas_cumprod = (1 - 2 * s) * f + s
    return alphas_cumprod


def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule, as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    t = np.linspace(1, T, T)
    f = np.cos(((t / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    return clip_noise_schedule(alphas_cumprod, clip_margin=0.001)


class PositiveLinear(nn.Module):
    """Linear layer with weights forced to be positive.
    """

    def __init__(self, in_features, out_features, bias=True, weight_init_offset=-2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight_init_offset = weight_init_offset

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        positive_weight = F.softplus(self.weight)
        return F.linear(inputs, positive_weight, self.bias)


class BaseNoiseSchedule(nn.Module):

    @abc.abstractmethod
    def forward(self, t):
        raise NotImplementedError()

    def sweep(self, num_steps):
        t = torch.linspace(1, num_steps, num_steps) / num_steps
        gamma = self.forward(t.unsqueeze(-1)).squeeze(-1)
        return t, gamma


class FixedNoiseSchedule(BaseNoiseSchedule):
    """Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, shape, timesteps, precision):
        super().__init__()

        self.timesteps = timesteps

        if shape == "cosine":
            alphas_cumprod = cosine_beta_schedule(timesteps, s=precision)

        elif shape.startswith("polynomial"):
            splits = shape.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas_cumprod = polynomial_schedule(timesteps, s=precision, power=power)

        else:
            raise ValueError()

        alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        gamma = torch.log(1.0 - alphas_cumprod) - torch.log(alphas_cumprod)
        gamma = F.pad(gamma, (1, 0), value=-torch.inf)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class LearnedNoiseSchedule(BaseNoiseSchedule):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper.
    """

    def __init__(self, d_hidden=1024):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, d_hidden)
        self.l3 = PositiveLinear(d_hidden, 1)

        self.gamma_0 = nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.0]))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)

        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        # Set t = 0 to -inf
        gamma = torch.where(t == 0.0, -torch.inf, gamma)

        return gamma
