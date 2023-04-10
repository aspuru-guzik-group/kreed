from typing import Literal

import dgl
import numpy as np
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from src.diffusion.dynamics import EquivariantDynamics
from src.modules import NoiseSchedule

from tqdm import tqdm

class EquivariantDDPMConfig(pydantic.BaseModel):
    """Configuration object for the DDPM."""

    equivariance: Literal["e3", "reflect"] = "reflect"

    # ===========
    # EGNN Fields
    # ===========

    d_egnn_atom_vocab: int = 16
    d_egnn_hidden: int = 256
    n_egnn_layers: int = 6

    # ===============
    # Sampling Fields
    # ===============

    timesteps: int = 1000
    noise_shape: str = "polynomial_2"
    noise_precision: float = 1e-5

    guidance_strength: float = 1.0


class EquivariantDDPM(nn.Module):

    def __init__(self, config: EquivariantDDPMConfig):
        super().__init__()

        self.config = config
        cfg = config

        self.dynamics = EquivariantDynamics(
            equivariance=cfg.equivariance,
            d_atom_vocab=cfg.d_egnn_atom_vocab,
            d_hidden=cfg.d_egnn_hidden,
            n_layers=cfg.n_egnn_layers,
        )

        self.T = cfg.timesteps
        self.gamma = NoiseSchedule(cfg.noise_shape, timesteps=self.T, precision=cfg.noise_precision)

    def dimensionality(self, G):
        # always subspace where weighted center of mass is 0
        return (G.batch_num_nodes() - 1) * 3

    def broadcast_scalar(self, val, G=None):
        if G is None:
            return val
        assert val.ndim == 1, val.shape
        return dgl.broadcast_nodes(G, val).unsqueeze(-1)

    def sigma(self, gamma, broadcast_to=None):
        sigma = torch.sigmoid(gamma).sqrt()
        return self.broadcast_scalar(sigma, G=broadcast_to)

    def alpha(self, gamma, broadcast_to=None):
        alpha = torch.sigmoid(-gamma).sqrt()
        return self.broadcast_scalar(alpha, G=broadcast_to)

    @staticmethod
    def SNR(gamma):
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s, broadcast_to=None):
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        values = (sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s)
        return tuple(self.broadcast_scalar(x, G=broadcast_to) for x in values)

    def sample_randn_G_like(self, G_init, mean=None, std=None, return_noise=False):
        eps = torch.randn_like(G_init.ndata["xyz"])
        eps = utils.zeroed_weighted_com(G_init, eps)

        G = G_init.local_var()
        if (mean is None) and (std is None):
            G.ndata["xyz"] = eps
        else:
            assert (mean is not None) and (std is not None)
            G.ndata["xyz"] = mean + (std * eps)
        
        # zero out nans
        if torch.isnan(G.ndata['xyz']).any():
            print('nan detected in sampling')
            G.ndata['xyz'] = torch.where(torch.isnan(G.ndata['xyz']), 0.0, G.ndata['xyz'])
                
        utils.assert_zeroed_weighted_com(G)

        return G if not return_noise else (G, eps)

    def sample_p_Gs_given_Gt(self, G_t, s, t):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, broadcast_to=G_t)
        )

        sigma_s = self.sigma(gamma_s, broadcast_to=G_t)
        sigma_t = self.sigma(gamma_t, broadcast_to=G_t)

        eps_t = self.dynamics(G=G_t, t=(t.float() / self.T))
        mu = (G_t.ndata["xyz"] / alpha_t_given_s) - ((sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        return self.sample_randn_G_like(G_t, mean=mu, std=sigma)

    def sample_p_G_given_G0(self, G_0):
        zeros = torch.zeros([G_0.batch_size], dtype=torch.int, device=G_0.device)
        gamma_0 = self.gamma(zeros)

        sigma_0 = self.sigma(gamma_0, broadcast_to=G_0)
        alpha_0 = self.alpha(gamma_0, broadcast_to=G_0)

        eps = self.dynamics(G_0, t=zeros.float())
        mu = 1.0 / alpha_0 * (G_0.ndata["xyz"] - sigma_0 * eps)
        sigma = self.broadcast_scalar(self.SNR(-0.5 * gamma_0), G=G_0)

        return self.sample_randn_G_like(G_0, mean=mu, std=sigma)

    @torch.no_grad()
    def sample_p_G(self, G_init, keep_frames=None):
        G_T = self.sample_randn_G_like(G_init)
        utils.assert_zeroed_weighted_com(G_T)

        frames = {self.T: G_T}

        G_t = G_T
        for step in tqdm(reversed(range(0, self.T)), desc='Sampling', leave=False, total=self.T):
            s = torch.full([G_T.batch_size], fill_value=step, device=G_T.device)
            G_t = self.sample_p_Gs_given_Gt(G_t=G_t, s=s, t=(s + 1))
            if (keep_frames is not None) and (step in keep_frames):
                frames[step] = G_t
        G = self.sample_p_G_given_G0(G_t)
        utils.assert_zeroed_weighted_com(G)  # sanity check

        frames[-1] = G

        if keep_frames is None:
            return G
        else:
            return G, frames

    def mse_error(self, G, input, target, reduction):
        with G.local_scope():
            G.ndata["error"] = (input - target).square()
            if reduction == "sum":
                error = dgl.sum_nodes(G, "error").sum(dim=-1)
            elif reduction == "mean":
                error = dgl.mean_nodes(G, "error").mean(dim=-1)
            else:
                raise ValueError()
        return error

    def simple_losses(self, G):
        t = torch.randint(0, self.T + 1, size=[G.batch_size], dtype=torch.int, device=G.device)

        gamma_t = self.gamma(t)
        alpha_t = self.alpha(gamma_t, broadcast_to=G)
        sigma_t = self.sigma(gamma_t, broadcast_to=G)

        mu_t = alpha_t * G.ndata["xyz"]
        G_t, eps_true = self.sample_randn_G_like(G, mean=mu_t, std=sigma_t, return_noise=True)
        eps_pred = self.dynamics(G=G_t, t=(t.float() / self.T))

        return self.mse_error(G=G_t, input=eps_pred, target=eps_true, reduction="mean")

    def log_norm_const_p_G_given_G0(self, G):
        zeros = torch.zeros([G.batch_size], dtype=torch.int, device=G.device)
        gamma_0 = self.gamma(zeros)

        d = self.dimensionality(G)
        return -0.5 * d * (np.log(2 * np.pi) + gamma_0)

    def prior_matching_loss(self, G):
        T = torch.full([G.batch_size], fill_value=self.T, device=G.device)

        gamma_T = self.gamma(T)
        alpha_T = self.alpha(gamma_T, broadcast_to=G)

        mu_T = alpha_T * G.ndata["xyz"]
        sigma_T = self.sigma(gamma_T)

        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(sigma_T)
        return utils.gaussian_KL_div(G, mu_T, sigma_T, zeros, ones, d=self.dimensionality(G))

    def nlls(self, G):
        t = torch.randint(1, self.T + 1, size=[G.batch_size], device=G.device)
        s = t - 1

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        alpha_t = self.alpha(gamma_t, broadcast_to=G)
        sigma_t = self.sigma(gamma_t, broadcast_to=G)

        mu_t = alpha_t * G.ndata["xyz"]
        G_t, eps_true = self.sample_randn_G_like(G, mean=mu_t, std=sigma_t, return_noise=True)
        eps_pred = self.dynamics(G=G_t, t=(t.float() / self.T))

        SNR_weight = self.SNR(gamma_s - gamma_t) - 1
        error = self.mse_error(G=G_t, input=eps_pred, target=eps_true, reduction="sum")
        loss_tge0 = 0.5 * SNR_weight * error

        loss_prior = self.prior_matching_loss(G)

        zeros = torch.zeros_like(s)
        gamma_0 = self.gamma(zeros)
        alpha_0 = self.alpha(gamma_0, broadcast_to=G)
        sigma_0 = self.sigma(gamma_0, broadcast_to=G)

        mu_0 = alpha_0 * G.ndata["xyz"]
        G_0, eps_true = self.sample_randn_G_like(G, mean=mu_0, std=sigma_0, return_noise=True)
        eps_pred = self.dynamics(G=G_0, t=zeros.float())

        error = self.mse_error(G=G_0, input=eps_pred, target=eps_true, reduction="sum")
        loss_0 = 0.5 * error

        return loss_prior + (self.T * loss_tge0) + loss_0 - self.log_norm_const_p_G_given_G0(G)
