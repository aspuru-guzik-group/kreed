import math

import dgl
import numpy as np
import torch
from torch.nn import functional as F

import src.diffusion.distributions as dists
from src.diffusion.schedules import LearnedNoiseSchedule, FixedNoiseSchedule


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma ** 2) + 0.5
    )


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def alphas_cumprod(gamma):
    return torch.sigmoid(-gamma)


def SNR(gamma):
    return torch.exp(-gamma)


class EnVariationalDiffusion(torch.nn.Module):
    """The E(n) Diffusion Module.
    """

    def __init__(
        self,
        dynamics,
        timesteps=1000,
        noise_shape="learned",
        noise_precision=1e-4,
        loss_type="VLB",
        lamb_hybrid=0.001,
    ):
        super().__init__()

        assert loss_type in {"VLB", "L2", "hybrid"}

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
            step = torch.full([G.bach_size, 1], step, device=G.device)
        else:
            assert step.shape == (G.batch_size, 1)
        return step

    def dist_q_Gt_given_Gs(self, G_s, s, t):
        """Computes the mean and variance of q(G_t|G_s) where s < t."""

        t = self.broadcast_step(G_s, step=t)
        gamma_t = self.gamma(t)
        alphas_cumprod_t = alphas_cumprod(self.gamma(t))

        if s == 0:
            alphas_prod = alphas_cumprod(self.gamma(t))
        else:
            s = self.broadcast_step(G_s, step=s)
            gamma_s = self.gamma(s)
            alphas_prod = torch.exp(F.logsigmoid(-gamma_t) - F.logsigmoid(-gamma_s))

        mean_t = dgl.broadcast_nodes(alphas_prod.sqrt(), G_s) * G_s.ndata["xyz"]  # (n_nodes, 3)
        var_t = 1.0 - alphas_cumprod_t.squeeze(-1)  # (batch_size)
        return mean_t, var_t

    def sample_randn_G_like(self, G_init, mean=None, var=None, tie_noise=False, return_noise=False):
        """Samples from G ~ p(G_base), where p = N(mean, var), or p = N(0, I) if mean and var are not given."""

        if tie_noise:
            n_nodes = torch.unique(G_init.batch_num_nodes()).shape
            assert n_nodes.numel() == 1  # only 1 unique graph size
            eps = torch.randn((n_nodes.item(), 3))
            eps = torch.cat([eps] * G_init.batch_size, dim=0)
        else:
            eps = torch.randn_like(G_init.ndata["xyz"])
        eps = dists.centered_mean(G_init, eps)

        G = G_init.local_var()
        if (mean is None) and (var is None):
            G.ndata["xyz"] = eps
        else:
            assert (mean is not None) and (var is not None)
            std = dgl.broadcast_nodes(G, var.sqrt())
            std = torch.broadcast_to(std, mean.shape)
            G.ndata["xyz"] = mean + std * eps
        return G if not return_noise else (G, eps)

    def sample_q_Gt_given_Gs(self, G_s, s, t, return_noise=False):
        """Samples from G_t ~ q(G_t|G_s), where s < t."""

        mu_t, var_t = self.dist_q_Gt_given_G0(G_s=G_s, s=s, t=t)
        return self.sample_randn_G_like(G_init=G_s, mean=mu_t, var=var_t, tie_noise=False, return_noise=return_noise)

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

        # sanity check
        dists.assert_centered_mean(G_t, G_t.ndata["xyz"])
        dists.assert_centered_mean(G_t, noise)

        mean = (G_t.ndata["xyz"] - dgl.broadcast_nodes(G_t, scale) * noise) / dgl.broadcast_nodes(G_t, alphas_t.sqrt())
        if last_step:
            var = SNR(-0.5 * gamma_t)
        else:
            var = (1.0 - alphas_t) * (1.0 - alphas_cumprod_tm1) / (1 - alphas_cumprod_t)

        # sample next coordinates
        return self.sample_randn_G_like(G_init=G_t, mean=mean, var=var, tie_noise=tie_noise)

    @torch.no_grad()
    def sample_p_G0(self, G_init, tie_noise=False, keep_frames=None):
        """Draw samples from the generative model."""

        G_T = self.sample_randn_G_like(G_init, tie_noise=tie_noise)
        dists.assert_centered_mean(G_T, G_T.ndata["xyz"])

        frames = {self.T: G_T}

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        G_t = G_T
        for t in range(self.T, 0, -1):
            G_t = self.sample_p_Gtm1_given_Gt(G_t=G_t, t=t, tie_noise=tie_noise)
            if t in keep_frames:
                frames[t] = keep_frames
        dists.assert_centered_mean(G_t, G_t.ndata["xyz"])  # sanity check

        frames[0] = G_t

        if keep_frames is None:
            return G_t
        else:
            return G_t, keep_frames

    def l2_noise_error(self, G, pred_eps, true_eps, reduce_sum=True):
        with G.local_scope():
            G.ndata["error"] = (pred_eps - true_eps).square()
            if reduce_sum:
                error = dgl.sum_nodes(G, "error").sum(dim=-1)
            else:
                error = dgl.mean_nodes(G, "error").mean(dim=-1)
        return error

    def log_norm_const_p_G0_given_G1(self, G_0):
        """Computes the log-normalizing constant of p(G_0|G_1)."""

        # Recall that var_x = (1 - bar{alpha}_1) / bar{alpha}_1 = 1/SNR(gamma_1) = 1/exp(-gamma_1).

        ones = torch.ones([G_0.batch_size, 1], device=G_0.device)
        log_var_1 = self.gamma(ones)
        d = (G_0.batch_num_nodes() - 1) * 3
        return -0.5 * d * (np.log(2 * np.pi) + log_var_1)

    def log_unnormalized_p_G0_given_G1(self, G_0, gamma_0, eps, pred_eps, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.beta(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.l2_noise_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def prior_matching_loss(self, G_0):
        """Computes KL[q(G_T|G_0)||p(G_0)] where p(G_0) ~ Normal(0, I) is the prior."""

        mu_T, var_T = self.dist_q_Gt_given_G0(G_s=G_0, s=0, t=self.T)
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(var_T)
        kl_div = dists.subspace_gaussian_KL_div(G_0, mu_T, var_T, zeros, ones)
        return kl_div

    def forward(self, G_0):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        include_loss_0 = not self.training
        use_L2_loss = self.training and (self.loss_type == "L2")

        # This part is about whether to include loss term 0 always.
        if include_loss_0:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({2, ..., T})
            lowest_t = 2
        else:
            # estimator = loss_t,           where t ~ U({1, ..., T})
            lowest_t = 1

        # Sample a timestep t
        t_int = torch.randint(lowest_t, self.T + 1, size=[G_0.batch_size, 1], device=G_0.device)
        t = t_int / self.T
        t_is_one = (t_int == 1)  # Important to compute log p(x | z0).

        gamma_t = self.gamma(t)
        gamma_tm1 = self.gamma(t - (1 / self.T))

        # sample G_t
        G_t, eps = self.sample_q_Gt_given_Gs(G_s=G_0, s=0, t=t, return_noise=True)
        dists.assert_centered_mean(G_t, G_t.ndata["xyz"])

        # Neural net prediction.
        pred_eps = self.phi(G_t=G_t, t=t)

        # Compute the error.
        error = self.l2_noise_error(G=G_t, pred_eps=pred_eps, true_eps=eps, reduce_sum=(not self.training))

        if use_L2_loss:
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_tm1 - gamma_t) - 1).squeeze(-1)
        loss_t = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_norm_const_p_G0_given_G1(G_0=G_0)

        # Reset constants during training with l2 loss.
        if use_L2_loss:
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        loss_prior = self.prior_matching_loss(G_0=G_0)

        # Combining the terms
        if include_loss_0:
            loss_t = loss_t
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.beta(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            pred_eps = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_recons = -self.log_unnormalized_p_G0_given_G1(
                x, h, z_0, gamma_0, eps_0, pred_eps, node_mask)

            assert loss_prior.size() == estimator_loss_terms.size()
            assert loss_prior.size() == neg_log_constants.size()
            assert loss_prior.size() == loss_recons.size()

            loss = loss_prior + estimator_loss_terms + neg_log_constants + loss_recons

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_recons = -self.log_unnormalized_p_G0_given_G1(G_0=G_0, gamma_0=gamma_t, eps=eps, pred_eps=pred_eps)
            loss_t = torch.where(t_is_one, loss_recons, loss_t)

            # Only upweigh estimator if using the vlb objective.
            if use_L2_loss:
                loss_t_scale = 1
            else:
                loss_t_scale = self.T

            assert loss_prior.size() == loss_t.size()
            assert loss_prior.size() == neg_log_constants.size()

            loss = loss_prior + (loss_t_scale * loss_t) + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {
            't': t.squeeze(),
            'loss_t': loss.squeeze(),
            'error': error.squeeze()
        }
