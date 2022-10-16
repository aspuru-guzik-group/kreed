import math

import dgl
import numpy as np
import torch
from equivariant_diffusion import utils as diffusion_utils
from torch.nn import functional as F

import src.diffusion.distributions as dists
from src.diffusion.schedules import LearnedNoiseSchedule, FixedNoiseSchedule


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma ** 2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2)
            - 0.5
        ) * node_mask
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
        loss_type="vlb",
        lamb_hybrid=0.001,
    ):
        super().__init__()

        assert loss_type in {"vlb", "l2", "hybrid"}

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.T = timesteps
        self.loss_type = loss_type

        if noise_shape == "learned":
            assert (loss_type == "vlb"), "A noise schedule can only be learned with a vlb objective."
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

    def dist_q_Gs_given_Gt(self, G_t, t, s):
        """Computes the mean and variance of q(G_t|G_s) where t > s."""

        t = self.broadcast_step(G_t, step=t)
        s = self.broadcast_step(G_t, step=s)

        alphas_cumprod_t = alphas_cumprod(self.gamma(t))
        alphas_cumprod_s = alphas_cumprod(self.gamma(s))

        alpha_prod_stot = torch.exp(torch.log(alphas_cumprod_t) - torch.log(alphas_cumprod_s))

    def dist_q_Gt_given_G0(self, G_0, t):
        """Computes the mean and variance of q(G_t|G_0)."""

        t = self.broadcast_step(G=G_0, step=t)

        gamma_t = self.gamma(t)
        alphas_cumprod_t = alphas_cumprod(gamma_t)
        mean_scale = alphas_cumprod_t.sqrt()

        mean_t = dgl.broadcast_nodes(mean_scale, G_0) * G_0.ndata["xyz"]  # (n_nodes, 3)
        var_t = 1.0 - alphas_cumprod_t.squeeze(-1)  # (batch_size)
        return mean_t, var_t

    def pred_G0_from_Gt(self, G_t, eps_t, gamma_t):
        z_t = G_t.ndata["xyz"]
        alphas_cumprod_t = alphas_cumprod(gamma_t)
        alphas_cumprod_t = dgl.broadcast_nodes(alphas_cumprod_t, G_t)

        G_0 = G_t.local_var()
        G_0.ndata["xyz"] = (z_t - (1.0 - alphas_cumprod_t).sqrt() * eps_t) / alphas_cumprod_t.sqrt()
        return G_0

    def compute_error(self, G, pred_eps, eps):
        with G.local_scope():
            G.ndata["error"] = (pred_eps - eps).square()
            if self.training and (self.loss_type == "l2"):
                error = dgl.mean_nodes(G, "error")
            else:
                error = dgl.sum_nodes(G, "error")
        return error

    def log_constants_p_x_given_z0(self, G, node_mask):
        """Computes p(x|z0)."""

        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (G.batch_num_nodes() - 1) * 3

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_pxh_given_z0_without_constants(
        self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10
    ):
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
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

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

        mu_T, var_T = self.dist_q_Gt_given_G0(G_0=G_0, t=self.T)
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(var_T)

        d = (G_0.batch_num_nodes() - 1) * 3
        kl_div = dists.gaussian_KL_divergence(G_0, mu_T, var_T, zeros, ones, d=d)
        return kl_div

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.beta(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.prior_matching_loss(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
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

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {
            't': t_int.squeeze(), 'loss_t': loss.squeeze(),
            'error': error.squeeze()
        }

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL."""

        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_GT_like(self, G, tie_noise):
        if tie_noise:
            n_nodes = torch.unique(G.batch_num_nodes()).shape
            assert n_nodes.numel() == 1  # only 1 unique graph size
            eps = torch.randn((n_nodes.item(), 3))
            eps = torch.cat([eps] * G.batch_size, dim=0)
        else:
            eps = torch.randn_like(G.ndata["xyz"])

        G_T = G.local_var()
        G_T.ndata["xyz"] = dists.centered_mean(G, eps)
        return G_T

    def sample_p_Gtm1_given_Gt(self, G_t, t, tie_noise=False):
        """Samples from G_s ~ p(G_s|G_t). Only used during sampling."""

        t = self.broadcast_step(G=G_t, step=t)
        tm1 = t - (1 / self.T)  # t - 1

        alphas_cumprod_t = alphas_cumprod(self.gamma(t))
        alphas_cumprod_tm1 = alphas_cumprod(self.gamma(tm1))
        alphas_t = alphas_cumprod_t / alphas_cumprod_tm1

        scale = (1.0 - alphas_t) / (1.0 - alphas_cumprod_t).sqrt()
        noise = self.dynamics(G=G_t, t=t)

        # sanity check
        dists.assert_centered_mean(G_t, G_t.ndata["xyz"])
        dists.assert_centered_mean(G_t, noise)

        mean = (G_t.ndata["xyz"] + dgl.broadcast_nodes(G_t, scale) * noise)
        mean = mean / dgl.broadcast_nodes(G_t, alphas_t.sqrt())

        var = (1.0 - alphas_t) * (1.0 - alphas_cumprod_tm1) / (1 - alphas_cumprod_t)
        std = dgl.broadcast_nodes(G_t, var.sqrt())
        std = torch.broadcast_to(std, mean.shape)

        # sample next coordinates
        G_tm1 = self.sample_GT_like(G_t, tie_noise=tie_noise)
        G_tm1.ndata["xyz"] = mean + std * G_tm1.ndata["xyz"]
        dists.assert_centered_mean(G_tm1, G_tm1.ndata["xyz"])  # sanity check
        return G_tm1

    @torch.no_grad()
    def sample(self, G_init, tie_noise=False, keep_frames=None):
        """Draw samples from the generative model."""

        G_T = self.sample_GT_like(G_init, tie_noise=tie_noise)
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
