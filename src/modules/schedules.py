import abc

import numpy as np
import torch
import torch.nn as nn


def clip_noise_schedule(alphas2, margin=0.001):
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=margin, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(timesteps, s=1e-5, power=2.0):
    T = timesteps + 1
    t = np.linspace(0, T, T)
    alphas2 = (1 - np.power(t / T, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, margin=0.001)
    return (1 - 2 * s) * alphas2 + s


class BaseNoiseSchedule(nn.Module):

    @abc.abstractmethod
    def forward(self, t):
        raise NotImplementedError()

    def sweep(self, num_steps):
        t = torch.linspace(1, num_steps, num_steps) / num_steps
        gamma = self.forward(t.unsqueeze(-1)).squeeze(-1)
        return t, gamma


class FixedNoiseSchedule(BaseNoiseSchedule):

    def __init__(self, shape, timesteps, precision):
        super().__init__()

        self.timesteps = timesteps

        if shape.startswith("polynomial"):
            splits = shape.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError()

        alphas2 = torch.from_numpy(alphas2)
        gammas = torch.log(1.0 - alphas2) - torch.log(alphas2)
        self.register_buffer("gammas", gammas.float())

    def forward(self, t):
        assert t.dtype == t.long()
        return self.gamma[t]
