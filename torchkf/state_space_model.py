from abc import ABC, abstractmethod
from typing import Optional, Union
from collections import OrderedDict

import torch.autograd
import torch.distributions as td
from torch.distributions import Transform, constraints

from .transformations import GaussianTransform, LinearTransform, LinearizedTransform, UnscentedTransform, Gaussian


class GaussianStateSpaceModel:
    def __init__(self, fwd_transform: GaussianTransform, obs_transform: GaussianTransform, state_dim: int, obs_dim: int,
                 initial_state_mean=None, initial_state_cov=None, process_noise_mean=None, process_noise_cov=None,
                 observation_noise_mean=None, observation_noise_cov=None):
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        if initial_state_mean is None:
            initial_state_mean = torch.zeros((state_dim,))
        if initial_state_cov is None:
            initial_state_cov = torch.eye(state_dim)
        if process_noise_mean is None:
            process_noise_mean = torch.zeros((state_dim,))
        if process_noise_cov is None:
            process_noise_cov = torch.eye(state_dim)
        if observation_noise_mean is None:
            observation_noise_mean = torch.zeros((obs_dim,))
        if observation_noise_cov is None:
            observation_noise_cov = torch.eye(obs_dim)

        self._parameters = dict(
            initial_state_mean=initial_state_mean,
            initial_state_cov=initial_state_cov,
            process_noise_mean=process_noise_mean,
            process_noise_cov=process_noise_cov,
            observation_noise_mean=observation_noise_mean,
            observation_noise_cov=observation_noise_cov
        )

        self.fwd_transform = fwd_transform
        self.obs_transform = obs_transform

    def prepare_inputs(self, y):
        """ y: torch tensor ([n_batchs,] n_times, obs_dim)"""
        if not y.shape[-1] == self._obs_dim:
            raise ValueError(
                f"Dimension of measurements y ({y.shape[-1]} does not match declared obs_dim ({self._obs_dim}).")
        if len(y.shape) == 2:
            y = y.unsqueeze(0)
        elif len(y.shape) != 3:
            raise ValueError(
                f"Expected measurements y to have 2 or 3 dimensions, but got ({len(y.shape)}.")
        return y

    def filter(self, y: torch.Tensor, backward_pass=False):
        y = self.prepare_inputs(y)
        n_batchs, n_times, _ = y.shape

        process_noise = Gaussian(self._parameters['process_noise_mean'],
                                 self._parameters['process_noise_cov']).expand(torch.Size([n_batchs]))
        observation_noise = Gaussian(self._parameters['observation_noise_mean'],
                                     self._parameters['observation_noise_cov']).expand(torch.Size([n_batchs]))
        x_prev = Gaussian(self._parameters['initial_state_mean'],
                          self._parameters['initial_state_cov']).expand(torch.Size([n_batchs]))

        trajectory = []
        for i in range(n_times):
            x_prior, Px_post_x_prior = self.fwd_transform.apply(x_prev, full=True)
            x_prior = x_prior + process_noise

            y_prior, Pxy = self.obs_transform.apply(x_prior, full=True)
            y_prior = y_prior + observation_noise

            x_post = Gaussian.conditional(x_prior, y_prior, y[:, i], Pxy)

            trajectory.append(
                OrderedDict(x_prev=x_prev, x_prior=x_prior, x_post=x_post, y_prior=y_prior, Px_post_x_prior=Px_post_x_prior, Pxy=Pxy)
            )

            x_prev = x_post

        if backward_pass:
            x_prev, x_prior, x_post, y_prior, Px_post_x_prior, Pxy = trajectory[-1].values()
            x_backward = x_post
            for i in range(n_times):
                J = torch.bmm(Px_post_x_prior, x_prior.covariance_matrix.inverse())
                x_backward_new_mean = x_post.mean + torch.bmm(J, (x_backward.mean - x_prior.mean).unsqueeze(-1)).squeeze(-1)
                x_backward_new_cov = x_post.covariance_matrix + \
                                     torch.bmm(torch.bmm(J, (x_backward.covariance_matrix - x_prior.covariance_matrix)),  J.swapaxes(1,2))

                x_backward = Gaussian(x_backward_new_mean, x_backward_new_cov)
                trajectory[-i-1]['x_backward'] = x_backward

        return trajectory

