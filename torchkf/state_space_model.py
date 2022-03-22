from abc import ABC, abstractmethod
from typing import Optional, Union
from collections import OrderedDict

import torch.autograd
import torch.distributions as td
from torch.distributions import Transform, constraints

from .transformations import GaussianTransform, LinearTransform, LinearizedTransform, UnscentedTransform, Gaussian, stack_distributions


class GaussianStateSpaceModel:
    def __init__(self, fwd_transform: GaussianTransform, obs_transform: GaussianTransform, state_dim: int, obs_dim: int,
                 input_transform: Optional[GaussianTransform] = None, input_dim: int = 0,
                 initial_state_mean=None, initial_state_cov=None, process_noise_cov=None, observation_noise_cov=None):
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._input_dim = input_dim
        if initial_state_mean is None:
            initial_state_mean = torch.zeros((state_dim,))
        if initial_state_cov is None:
            initial_state_cov = torch.eye(state_dim)
        if process_noise_cov is None:
            process_noise_cov = torch.eye(state_dim)
        if observation_noise_cov is None:
            observation_noise_cov = torch.eye(obs_dim)

        self._parameters = dict(
            initial_state_mean=initial_state_mean,
            initial_state_cov=initial_state_cov,
            process_noise_cov=process_noise_cov,
            observation_noise_cov=observation_noise_cov
        )

        self.fwd_transform = fwd_transform
        self.obs_transform = obs_transform
        self.input_transform = input_transform

    def prepare_data(self, y):
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
        y = self.prepare_data(y)
        n_batchs, n_times, _ = y.shape

        process_noise = Gaussian(torch.zeros(self._state_dim), self._parameters['process_noise_cov']).expand(torch.Size([n_batchs]))
        observation_noise = Gaussian(torch.zeros(self._obs_dim), self._parameters['observation_noise_cov']).expand(torch.Size([n_batchs]))
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
            # x(T|T) = x(post)
            x_prev, x_prior, x_post, y_prior, Px_post_x_prior, Pxy = trajectory[-1].values()
            x_backward = x_post
            for i in range(n_times):
                # x(T-i-1|T-i-1), x(T-i|T-i-1),  x(T-i|T-i), Cov(x(T-i-1|T-i-1), x(T-i|T-i-1))
                x_prev, x_prior, x_post, y_prior, Px_post_x_prior, Pxy = trajectory[-i-1].values()  # xT

                # Jt = Cov(x(T-i-1|T-i-1), x(T-i|T-i-1)) * inverse( Cov(x(T-i|T-i-1)) )
                J = torch.bmm(Px_post_x_prior, x_prior.covariance_matrix.inverse())

                # x(T-i-1|T) = x(T-i-1|T-i-1) + Jt * (x(T-i|T) - x(T-i|T-i-1))
                x_backward_new_mean = x_prev.mean + torch.bmm(J, (x_backward.mean - x_prior.mean).unsqueeze(-1)).squeeze(-1)
                x_backward_new_cov = x_prev.covariance_matrix + \
                                     torch.bmm(torch.bmm(J, (x_backward.covariance_matrix - x_prior.covariance_matrix)), J.swapaxes(1,2))
                x_backward_new_cov = 0.5 * (x_backward_new_cov + x_backward_new_cov.swapaxes(1,2))

                x_backward = Gaussian(x_backward_new_mean, x_backward_new_cov)
                trajectory[-i-1]['x_backward'] = x_backward

        trajectory = OrderedDict(**{k: [i[k] for i in trajectory] for k in trajectory[0].keys()})
        for k, traj in trajectory.items():
            if isinstance(traj[0], Gaussian):
                trajectory[k] = stack_distributions(traj, dim=1)
            else:
                trajectory[k] = torch.stack(traj, dim=1)

        return trajectory

    def complete_data_likelihood(self, y, trajectory=None):
        if trajectory is None:
            trajectory = self.filter(y, backward_pass=True)

        x_ref = trajectory['x_backward'] if 'x_backward' in trajectory.keys() else trajectory['x_post']

        ll_x = 0
        ll_y = 0

        for i in range(x_ref.batch_shape[1]):
            if i > 0:
                x_pred = self.fwd_transform.call(x_ref[:, i-1].mean)
                ll_x += Gaussian(x_pred, self._parameters['process_noise_cov']).log_prob(x_ref[:, i].mean)
            y_pred = self.obs_transform.call(x_ref[:, i].mean)
            ll_y += Gaussian(y_pred, self._parameters['observation_noise_cov']).log_prob(y[:, i])

        return dict(ll=ll_x+ll_y, ll_x=ll_x, ll_y=ll_y)

    def fit_params(self, y, trajectory):
        x_ref = trajectory['x_backward'] if 'x_backward' in trajectory.keys() else trajectory['x_post']

        res_x, res_y = [], []
        for i in range(x_ref.batch_shape[1]):
            if i > 0:
                x_pred = self.fwd_transform.call(x_ref[:, i - 1].mean)
                res_x.append(x_ref[:, i].mean - x_pred)
            y_pred = self.obs_transform.call(x_ref[:, i].mean)
            res_y.append(y[:, i] - y_pred)

        res_x = torch.concat(res_x, dim=0)
        res_y = torch.concat(res_y, dim=0)

        Q = res_x.T @ res_x / (res_x.shape[0])
        Q = 0.5 * (Q.T + Q)
        self._parameters['process_noise_cov'] = Q

        R = res_y.T @ res_y / (res_y.shape[0])
        R = 0.5 * (R.T + R)
        self._parameters['observation_noise_cov'] = R

        m0 = x_ref.mean[:,0].mean(0)
        P0 = (x_ref.mean[:, 0] - m0[None]).T @ (x_ref.mean[:, 0] - m0[None])
        P0 += x_ref.covariance_matrix[:, 0].mean(0)
        P0 /= (x_ref.mean.shape[0])
        P0 = 0.5 * (P0 + P0.T)
        self._parameters['initial_state_mean'] = m0
        self._parameters['initial_state_cov'] = P0

    def blind_forecast(self, n_points):
        x = self._parameters['initial_state_mean']

        xs, ys = [], []

        for i in range(n_points):
            x_new = self.fwd_transform.call(x[None]).squeeze(0)
            y_new = self.obs_transform.call(x_new[None]).squeeze(0)

            xs.append(x_new)
            ys.append(y_new)

            x = x_new

        trajectory = dict()
        trajectory['x'] = Gaussian(torch.stack(xs, dim=0),
                                   self._parameters['process_noise_cov'].expand((n_points, self._state_dim, self._state_dim)))

        trajectory['y'] = Gaussian(torch.stack(ys, dim=0),
                                   self._parameters['observation_noise_cov'].expand((n_points, self._obs_dim, self._obs_dim)))

        return trajectory