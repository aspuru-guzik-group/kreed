import dgl
import numpy as np
import torch
from torch.nn import functional as F

import src.diffusion.distributions as dists
from src.diffusion.schedules import LearnedNoiseSchedule, FixedNoiseSchedule


def alphas_cumprod(gamma):
    return torch.sigmoid(-gamma)


def SNR(gamma):
    return torch.exp(-gamma)


class EnEquivariantDiffusionModel(torch.nn.Module):
    """The E(n) Diffusion Module.
    """

    def __init__(
        self,
        dynamics,
        timesteps=1000,
        noise_shape="polynomial_2",
        noise_precision=0.08,
        loss_type="L2",
    ):
        super().__init__()

        assert loss_type in {"VLB", "L2"}

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.T = timesteps
        self.loss_type = loss_type

        if noise_shape == "learned":
            assert (loss_type == "VLB"), "A noise schedule can only be learned with a vlb objective."
            self.gamma = LearnedNoiseSchedule()
        else:
            self.gamma = FixedNoiseSchedule(noise_shape, timesteps=self.T, precision=noise_precision)

    def broadcast_step(self, G, step):
        if isinstance(step, int):
            step = step / self.T  # normalize
        if isinstance(step, float):
            step = torch.full([G.batch_size, 1], step, device=G.device)
        else:
            assert step.shape == (G.batch_size, 1)
        return step


    def dist_q_Gt_given_Gs(self, G_s, s, t):
        """Computes the mean and variance of q(G_t|G_s) where s < t.
        and transition matrix Qs ... Qt"""

        t = self.broadcast_step(G_s, step=t)
        gamma_t = self.gamma(t)
        alphas_cumprod_t = alphas_cumprod(self.gamma(t))

        if s == 0:
            alphas_prod = alphas_cumprod(self.gamma(t))
        else:
            s = self.broadcast_step(G_s, step=s)
            gamma_s = self.gamma(s)
            alphas_prod = torch.exp(F.logsigmoid(-gamma_t) - F.logsigmoid(-gamma_s))

        mean_t = dgl.broadcast_nodes(G_s, alphas_prod.sqrt()) * G_s.ndata["xyz"]  # (n_nodes, 3)
        signs_01 = (G_s.ndata['signs']+1) / 2
        p_t = dgl.broadcast_nodes(G_s, alphas_prod) * signs_01 + (1-dgl.broadcast_nodes(G_s, alphas_prod)) / 2
        var_t = 1.0 - alphas_cumprod_t.squeeze(-1)  # (batch_size)

        return mean_t, var_t, p_t

    def sample_randn_G_like(self, G_init, mean=None, var=None, p=None, tie_noise=False, return_noise=False):
        """Samples from G ~ p(G_base), where p = N(mean, var), or p = N(0, I) if mean and var are not given."""
        # ignores abs nodes in mean and ignores free nodes in p

        if tie_noise:
            n_nodes = torch.unique(G_init.batch_num_nodes()).shape
            assert n_nodes.numel() == 1  # only 1 unique graph size
            eps = torch.randn((n_nodes.item(), 3))
            eps = torch.cat([eps] * G_init.batch_size, dim=0)
        else:
            eps = torch.randn_like(G_init.ndata["xyz"])
        # eps = dists.centered_mean(G_init, eps)
        eps = torch.where(G_init.ndata['free_mask'], eps, 0.0)

        G = G_init.local_var()
        if (mean is None) and (var is None) and (p is None):
            G.ndata["free_xyz"] = torch.where(G.ndata['free_mask'], eps, 0.0)
        else:
            assert (mean is not None) and (var is not None) and (p is not None)
            std = dgl.broadcast_nodes(G, var.sqrt())
            std = torch.broadcast_to(std.unsqueeze(-1), mean.shape)
            G.ndata["free_xyz"] = torch.where(G.ndata['free_mask'], mean + std * eps, 0.0) # ignore abs nodes
            
            new_signs_01 = torch.bernoulli(p)
            new_signs_p1m1 = 2*new_signs_01 - 1.0
            flipped_signs = G.ndata['signs'] * new_signs_p1m1
            G.ndata['signs'] = torch.where(G.ndata['abs_mask'], new_signs_p1m1, 0.0) # ignore free nodes
            G.ndata['xyz'] = G.ndata['signs'] * G.ndata['abs_xyz'] + G.ndata['free_xyz']
        return G if not return_noise else (G, eps, flipped_signs)

    def sample_q_Gt_given_Gs(self, G_s, s, t, return_noise=False):
        """Samples from G_t ~ q(G_t|G_s), where s < t."""

        mu_t, var_t, p_t = self.dist_q_Gt_given_Gs(G_s=G_s, s=s, t=t)
        return self.sample_randn_G_like(G_init=G_s, mean=mu_t, var=var_t, p=p_t, tie_noise=False, return_noise=return_noise)

    def sample_p_Gtm1_given_Gt(self, G_t, t: int, tie_noise=False):
        """Samples from G_{t-1} ~ p(G_{t-1}|G_t)."""

        last_step = (t == 1)

        t = self.broadcast_step(G=G_t, step=t)
        tm1 = t - (1 / self.T)  # t - 1

        gamma_t = self.gamma(t)
        alphas_cumprod_t = alphas_cumprod(gamma_t)
        alphas_cumprod_tm1 = alphas_cumprod(self.gamma(tm1))
        alphas_t = alphas_cumprod_t / alphas_cumprod_tm1

        scale = (1.0 - alphas_t) / (1.0 - alphas_cumprod_t).sqrt()
        noise = self.dynamics(G=G_t, t=t)
        p = torch.sigmoid(noise)

        mean = (G_t.ndata["xyz"] - dgl.broadcast_nodes(G_t, scale) * noise) / dgl.broadcast_nodes(G_t, alphas_t.sqrt())
        if last_step:
            var = SNR(-0.5 * gamma_t)
        else:
            var = (1.0 - alphas_t) * (1.0 - alphas_cumprod_tm1) / (1 - alphas_cumprod_t)
        var = var.squeeze(-1)

        # sample next coordinates
        return self.sample_randn_G_like(G_init=G_t, mean=mean, var=var, p=p, tie_noise=tie_noise)

    @torch.no_grad()
    def sample_p_G0(self, G_init, tie_noise=False, keep_frames=None):
        """Draw samples from the generative model."""

        G_T = self.sample_randn_G_like(G_init, tie_noise=tie_noise)

        frames = {self.T: G_T}

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        G_t = G_T
        for t in range(self.T, 0, -1):
            G_t = self.sample_p_Gtm1_given_Gt(G_t=G_t, t=t, tie_noise=tie_noise)
            if (keep_frames is not None) and (t in keep_frames):
                frames[t] = G_t

        frames[0] = G_t

        if keep_frames is None:
            return G_t
        else:
            return G_t, frames

    def noise_error(self, G, pred_eps, true_eps, reduction):
        with G.local_scope():
            G.ndata["error"] = torch.where(G.ndata['free_mask'], (pred_eps - true_eps).square(), 0.0)
            if reduction == "sum":
                error = dgl.sum_nodes(G, "error").sum(dim=-1)
            elif reduction == "mean":
                error = dgl.mean_nodes(G, "error").mean(dim=-1)
            else:
                error = G.ndata["error"]
        
        abs_mask = G.ndata['abs_mask']
        pred_sign_probs = torch.sigmoid(pred_eps[abs_mask])

        # for now, try to predict G_0's signs, rather than predicting which signs were flipped
        true_sign_targets_p1m1 = G.ndata['signs'][abs_mask]
        true_sign_targets_01 = (true_sign_targets_p1m1 + 1) / 2
        sign_bce = F.binary_cross_entropy(pred_sign_probs, true_sign_targets_01, reduction=reduction)

        # could weight the errors here
        return error + sign_bce

    def log_norm_const_p_G0_given_G1(self, G_0):
        """Computes the log-normalizing constant of p(G_0|G_1)."""

        # Recall that var_x = (1 - bar{alpha}_1) / bar{alpha}_1 = 1/SNR(gamma_1) = 1/exp(-gamma_1).
        ones = torch.ones([G_0.batch_size], device=G_0.device)
        log_var_1 = self.gamma(ones)
        d = G_0.ndata['free_mask'].sum()
        return -0.5 * d * (np.log(2 * np.pi) + log_var_1)

    def prior_matching_loss(self, G_0):
        """Computes KL[q(G_T|G_0)||p(G_0)] where p(G_0) ~ Normal(0, I) is the prior."""

        mu_T, var_T, p_T = self.dist_q_Gt_given_Gs(G_s=G_0, s=0, t=self.T)
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(var_T)

        kl_div = dists.subspace_gaussian_KL_div(G_0, mu_T, var_T, zeros, ones)

        abs_mask = G_0.ndata['abs_mask']

        bce = F.binary_cross_entropy(p_T[abs_mask], torch.ones_like(p_T[abs_mask])*.5, reduction='sum')

        return kl_div + bce

    def forward(self, G_0):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if self.training:
            # estimator = loss_tm1,           where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # loss_0 will be computed separately.
            # estimator = loss_0 + loss_tm1,  where t ~ U({2, ..., T})
            lowest_t = 2

        # Sample a timestep t
        t_int = torch.randint(lowest_t, self.T + 1, size=[G_0.batch_size, 1], device=G_0.device)
        t = t_int / self.T

        gamma_t = self.gamma(t)
        gamma_tm1 = self.gamma(t - (1 / self.T))

        SNR_weight = SNR(gamma_tm1 - gamma_t) - 1
        SNR_weight = torch.where(t_int == 1, 1, SNR_weight).squeeze(-1)  # w(0) = 1

        # sample G_t
        G_t, eps, flipped_signs = self.sample_q_Gt_given_Gs(G_s=G_0, s=0, t=t, return_noise=True)

        # Neural net prediction.
        pred_eps = self.dynamics(G=G_t, t=t)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        loss_prior = self.prior_matching_loss(G_0=G_0)

        # Combining the terms
        if self.training:

            use_l2 = (self.loss_type == "L2")

            error = self.noise_error(G_t, pred_eps=pred_eps, true_eps=eps, reduction="mean")

            loss_tm1 = 0.5 * (1.0 if use_l2 else SNR_weight) * error

            if use_l2:
                loss = loss_prior + loss_tm1
            else:
                neg_log_Z = -self.log_norm_const_p_G0_given_G1(G_0=G_0)
                loss = loss_prior + (self.T * loss_tm1) + neg_log_Z

        else:

            error = self.noise_error(G_t, pred_eps=pred_eps, true_eps=eps, reduction="sum")

            loss_tm1 = 0.5 * SNR_weight * error

            t_1 = torch.ones_like(t) / self.T
            G_1, eps, flipped_signs1 = self.sample_q_Gt_given_Gs(G_s=G_0, s=0, t=t_1, return_noise=True)
            pred_eps = self.dynamics(G=G_1, t=t_1)
            error = self.noise_error(G_t, pred_eps=pred_eps, true_eps=eps, reduction="sum")

            loss_0 = (0.5 * error) - self.log_norm_const_p_G0_given_G1(G_0=G_0)
            loss = loss_prior + ((self.T - 1) * loss_tm1) + loss_0

        return loss
