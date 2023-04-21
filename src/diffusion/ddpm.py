import functools
from typing import Literal

import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from src import utils
from src.diffusion.dynamics import DummyDynamics, EquivariantDynamics
from src.modules import NoiseSchedule, PositionalEmbedding


class EquivariantDDPMConfig(pydantic.BaseModel):
    """Configuration object for the DDPM."""

    # ============
    # Model Fields
    # ============

    architecture: Literal["dummy", "edm"] = "edm"
    parameterization: Literal["eps", "x"] = "eps"
    timestep_embedding: Literal["none", "positional"] = "positional"

    atom_features: int = 32
    temb_features: int = 128
    cond_features: int = 128
    hidden_features: int = 256

    num_layers: int = 6
    norm_type: Literal["layer", "graph", "none"] = "graph"
    norm_adaptively: bool = True
    act: Literal["silu", "gelu"] = "silu"

    egnn_equivariance: Literal["e3", "ref"] = "ref"
    egnn_relaxed: bool = True
    zero_com_before_blocks: bool = True

    # ===============
    # Sampling Fields
    # ===============

    timesteps: int = 1000
    noise_shape: str = "polynomial_2"
    noise_precision: float = 1e-5

    guidance_strength: float = 0.0


class EquivariantDDPM(nn.Module):

    def __init__(self, config: EquivariantDDPMConfig):
        super().__init__()

        self.config = config
        cfg = config

        if cfg.timestep_embedding == "positional":
            self.embed_timestep = PositionalEmbedding(cfg.temb_features)
        else:
            cfg.temb_features = 1
            self.embed_timestep = None

        if cfg.architecture == "dummy":
            self.dynamics = DummyDynamics()  # for debugging
        elif cfg.architecture == "edm":
            self.dynamics = EquivariantDynamics(**dict(cfg))
        else:
            raise ValueError()

        self.T = cfg.timesteps
        self.gamma = NoiseSchedule(cfg.noise_shape, timesteps=self.T, precision=cfg.noise_precision)

    def broadcast_scalar(self, M, t):
        t = t.view(-1, 1)
        if t.shape[0] == M.batch_size:
            t = M.broadcast(t)
        assert t.shape[0] == M.coords.shape[0]
        return t

    def forward(self, M, t, puncond=0.0):
        cfg = self.config

        if puncond > 0:
            uncond_mask = torch.rand([M.batch_size, 1]).to(M.coords) <= puncond
            uncond_mask = M.broadcast(uncond_mask)
            M = M.replace(
                cond_labels=torch.where(uncond_mask, 0.0, M.cond_labels),
                cond_mask=torch.where(uncond_mask, False, M.cond_labels),
                moments=torch.where(uncond_mask, 0.0, M.moments),
            )

        t = self.broadcast_scalar(M, t)
        if cfg.timestep_embedding == "none":
            temb = t.float() / self.T
        else:
            temb = self.embed_timestep(t)

        out = self.dynamics(M=M, temb=temb)

        if cfg.parameterization == "eps":
            return out - M.coords
        else:
            return out

    def guided_forward(self, M, t, w=None):
        w = self.config.guidance_strength if (w is None) else w
        if w == 0:
            return self(M=M, t=t)
        else:
            return ((1 + w) * self(M=M, t=t, puncond=0.0)) - (w * self(M=M, t=t, puncond=1.0))

    def sigma(self, gamma):
        return torch.sigmoid(gamma).sqrt()

    def alpha(self, gamma):
        return torch.sigmoid(-gamma).sqrt()

    def SNR(self, gamma):
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s):
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sample_M_randn_like(self, M, mean=None, std=None, return_noise=False):
        eps = torch.randn_like(M.coords)
        coords = eps if (mean is None) else (mean + std * eps)
        coords = utils.zeroed_com(M, coords, orthogonal=True)
        M = M.replace(coords=coords)
        return (M, eps) if return_noise else M

    def denoised_from_dynamics_out(self, M_t, t, out):
        t = self.broadcast_scalar(M_t, t)
        gamma_t = self.gamma(t)
        sigma_t = self.sigma(gamma_t)
        alpha_t = self.alpha(gamma_t)

        if self.config.parameterization == "eps":
            return 1.0 / alpha_t * (M_t.coords - sigma_t * out)
        elif self.config.parameterization == "x":
            return out
        else:
            raise ValueError()

    def sample_Ms_given_Mt(self, M_t, s, t, w=None):
        assert torch.all(s < t)
        s = self.broadcast_scalar(M_t, s)
        t = self.broadcast_scalar(M_t, t)

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        alpha_s = self.alpha(gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        out = self.guided_forward(M=M_t, t=t, w=w)
        out = self.denoised_from_dynamics_out(M_t=M_t, t=t, out=out)

        mu = ((alpha_t_given_s * sigma_s.square() * M_t.coords) + (alpha_s * sigma2_t_given_s * out)) / sigma_t.square()
        sigma = sigma_t_given_s * sigma_s / sigma_t
        return self.sample_M_randn_like(M_t, mean=mu, std=sigma)

    def sample_M_given_M0(self, M_0, w=None):
        zeros = torch.zeros([M_0.batch_size], dtype=torch.int, device=M_0.device)
        zeros = self.broadcast_scalar(M_0, zeros)

        out = self.guided_forward(M=M_0, t=zeros, w=w)
        mu = self.denoised_from_dynamics_out(M_t=M_0, t=zeros, out=out)
        sigma = self.SNR(-0.5 * self.gamma(zeros))
        return self.sample_M_randn_like(M_0, mean=mu, std=sigma)

    @torch.no_grad()
    def sample(self, M, keep_frames=None, w=None):
        M = M.replace(coords=torch.zeros_like(M.coords))  # safety, so we don't cheat
        M_T = self.sample_M_randn_like(M)
        frames = {self.T: M_T}

        M_t = M_T
        for step in tqdm.tqdm(reversed(range(0, self.T)), desc="Sampling", leave=False, total=self.T):
            s = torch.full(size=[M.batch_size], fill_value=step, device=M.device)
            M_t = self.sample_Ms_given_Mt(M_t=M_t, s=s, t=(s + 1), w=w)

            if not torch.isfinite(M_t.coords).all():
                print("NaNs, detected. Setting to 0")
                M_t.replace(coords=torch.zeros_like(M_t.coords))
            if (keep_frames is not None) and (step in keep_frames):
                frames[step] = M_t.cpu()

        M = self.sample_M_given_M0(M_t)
        utils.assert_zeroed_com(M, M.coords)
        frames[-1] = M.cpu()

        return M if (keep_frames is None) else (M, frames)

    def denoising_errors(self, forward_fn, M, t, reduction):
        t = self.broadcast_scalar(M, t)
        gamma_t = self.gamma(t)
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)

        M_t, eps = self.sample_M_randn_like(M, mean=(alpha_t * M.coords), std=sigma_t, return_noise=True)
        out = forward_fn(M=M_t, t=t)

        if self.config.parameterization == "eps":
            errors = (out - eps).square()
        elif self.config.parameterization == "x":
            errors = (out - M.coords).square()
        else:
            raise ValueError()

        if reduction == "mean":
            return M.mean_pool(errors).mean(dim=-1)
        elif reduction == "sum":
            return M.sum_pool(errors).sum(dim=-1)
        else:
            raise ValueError()

    def simple_losses(self, M, puncond=0.0):
        t = torch.randint(0, self.T + 1, size=[M.batch_size], device=M.device)
        forward_fn = functools.partial(self.forward, puncond=puncond)
        return self.denoising_errors(forward_fn, M=M, t=t, reduction="mean")

    def dimensionality(self, M):
        return (M.num_atoms - 1) * 3  # subspace where atom-weighted center of mass is 0

    def gaussian_KL_qp(self, M, q_mean, q_std, p_mean, p_std):
        assert q_std.ndim == p_std.ndim == 1
        d = self.dimensionality(M)
        mean_sqe_dist = M.sum_pool((q_mean - p_mean).square()).sum(dim=-1)
        kl_div = (2 * d * torch.log(p_std / q_std)) + ((d * q_std.square() + mean_sqe_dist) / p_std.square()) - d
        return 0.5 * kl_div

    def log_norm_const_p_M_given_M0(self, M):
        zeros = torch.zeros([M.batch_size], dtype=torch.int, device=M.device)
        gamma_0 = self.gamma(zeros)
        d = self.dimensionality(M)
        return -0.5 * d * (np.log(2 * np.pi) + gamma_0)

    def prior_matching_loss(self, M):
        T = torch.full(size=[M.batch_size], fill_value=self.T, device=M.device)
        gamma_T = self.gamma(T)
        alpha_T = self.broadcast_scalar(M, self.alpha(gamma_T))

        mu_T, sigma_T = (alpha_T * M.coords), self.sigma(gamma_T)
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(sigma_T)
        return self.gaussian_KL_qp(M, mu_T, sigma_T, zeros, ones)

    def nlls(self, M, w=None):
        forward_fn = functools.partial(self.guided_forward, w=w)

        t = torch.randint(1, self.T + 1, size=[M.batch_size], device=M.device)
        SNR_weight = self.SNR(self.gamma(t - 1) - self.gamma(t)) - 1

        loss_prior = self.prior_matching_loss(M)
        loss_tge0 = 0.5 * SNR_weight * self.denoising_errors(forward_fn, M=M, t=t, reduction="sum")
        loss_0 = 0.5 * self.denoising_errors(forward_fn, M=M, t=torch.zeros_like(t), reduction="sum")

        return loss_prior + (self.T * loss_tge0) + loss_0 - self.log_norm_const_p_M_given_M0(M)
