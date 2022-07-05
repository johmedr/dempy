import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, List
from collections import OrderedDict
from tqdm import tqdm

import torch.autograd
import torch.distributions as td
from torch.distributions import Transform, constraints

from .transformations import GaussianTransform, LinearTransform, LinearizedTransform, UnscentedTransform, Gaussian, stack_distributions


class GaussianSystem(dict):
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 fwd_transform: GaussianTransform,
                 obs_transform: GaussianTransform,
                 initial_state_mean: torch.Tensor,
                 initial_state_cov: torch.Tensor,
                 process_noise_cov: torch.Tensor,
                 obs_noise_cov: torch.Tensor,
                 input_dim: int=0):

        super().__init__(
            state_dim=state_dim,
            obs_dim=obs_dim,
            input_dim=input_dim,
            fwd_transform=fwd_transform,
            obs_transform=obs_transform,
            initial_state_mean=initial_state_mean,
            initial_state_cov=initial_state_cov,
            process_noise_cov=process_noise_cov,
            obs_noise_cov=obs_noise_cov)


class HierarchicalDynamicalModel:
    def __init__(self, systems: List[GaussianSystem], input_mean=None, input_cov=None):
        HierarchicalDynamicalModel.check_systems(systems)
        
        self._systems    = systems
        self._n_systems  = len(systems)
        self._input_dim  = systems[0]['input_dim']
        self._obs_dim    = systems[-1]['obs_dim']
        self._input_mean = input_mean
        self._input_cov  = input_cov

    @staticmethod
    def check_systems(systems):
        if len(systems) < 2:
            return

        for k in range(1, len(systems)):
            if not systems[k-1]['obs_dim'] == systems[k]['input_dim']:
                raise ValueError(f"Dimension mismatch between index {k}.obs_dim "
                                 f"({systems[k-1]['obs_dim']}) and index {k+1}.input_dim ({systems[k-1]['obs_dim']}) "
                                 f"(indexing from 1)")

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

        process_noises = [
            Gaussian.with_covariance_handle(torch.zeros(system['state_dim']),
                     system['process_noise_cov']).expand(torch.Size([n_batchs]))
            for system in self._systems
        ]

        observation_noises = [
            Gaussian.with_covariance_handle(torch.zeros(system['obs_dim']),
                     system['obs_noise_cov']).expand(torch.Size([n_batchs]))
            for system in self._systems
        ]

        x_prevs = [
            Gaussian.with_covariance_handle(system['initial_state_mean'],
                     system['initial_state_cov']).expand(torch.Size([n_batchs]))
            for system in self._systems
        ]

        trajectory = []
        for i in tqdm(range(n_times), desc="Filter"):
            # level_data stores all variables at each level
            level_data = []

            # Downwards predictions
            for k in range(self._n_systems):
                if k == 0: # For now no input at level 0
                    u_prior = None
                    Pu_x_prior = None
                    x_prior, Px_prev_x_prior = self._systems[k]['fwd_transform'].apply(x_prevs[k], full=True)

                else:
                    u_prior = level_data[k-1]['y_prior']

                    x_prior, Pfwd = self._systems[k]['fwd_transform'].apply(x_prevs[k], u_prior, full=True)
                    Px_prev_x_prior, Pu_x_prior = torch.split(Pfwd, [self._systems[k]['state_dim'],
                                                                     self._systems[k]['input_dim']], dim=-2)

                x_prior = x_prior + process_noises[k]

                y_prior, Pxy = self._systems[k]['obs_transform'].apply(x_prior, full=True)
                y_prior = y_prior + observation_noises[k]

                level_data.append(
                    OrderedDict(x_prev=x_prevs[k],
                                x_prior=x_prior,
                                y_prior=y_prior,
                                u_prior=u_prior,
                                Px_prev_x_prior=Px_prev_x_prior,
                                Pu_x_prior=Pu_x_prior,
                                Pxy=Pxy)
                )

            # Upwards updates
            for k in reversed(range(self._n_systems)):
                if k == self._n_systems - 1:
                    y_post = y[:, i]
                else:
                    y_post = level_data[k+1]['u_post']

                x_post = Gaussian.conditional(
                    level_data[k]['x_prior'],
                    level_data[k]['y_prior'],
                    y_post,
                    level_data[k]['Pxy'])

                if k == 0:
                    u_post = None
                else:
                    u_post = Gaussian.conditional(
                        level_data[k]['u_prior'],
                        level_data[k]['x_prior'],
                        x_post,
                        level_data[k]['Pu_x_prior']
                    )

                level_data[k]['x_post'] = x_post
                level_data[k]['u_post'] = u_post

            trajectory.append(
                level_data
            )

            x_prevs = [_['x_post'] for _ in level_data]
        #
        if backward_pass:
            x_backwards = [_['x_post'] for _ in trajectory[-1]]
            for i in tqdm(range(n_times - 1), desc='Smooth'):
                level_data = trajectory[-i-1]

                # Upward updates
                for k in reversed(range(self._n_systems)):

                    x_backward = Gaussian.conditional(
                        level_data[k]['x_prev'],
                        level_data[k]['x_prior'],
                        x_backwards[k],
                        level_data[k]['Px_prev_x_prior'],
                    )

                    if k == 0:
                        u_backward = None
                    else:
                        u_backward = Gaussian.conditional(
                            level_data[k]['u_prior'],
                            level_data[k]['x_prior'],
                            x_backwards[k],
                            level_data[k]['Pu_x_prior'],
                        )
                    trajectory[-i-1][k]['u_backward'] = u_backward
                    trajectory[-i-2][k]['x_backward'] = x_backward
                    if i == 0:
                        trajectory[-1][k]['x_backward'] = trajectory[-1][k]['x_post']

                x_backwards = [_['x_backward'] for _ in trajectory[-i-2]]

                    # elif k == 0:
                    #     x_backward = Gaussian.conditional(
                    #         level_data[k]['x_prev'],
                    #         level_data[k]['x_prior'],
                    #         x_backwards[k],
                    #         level_data[k]['Px_prev_x_prior'],
                    #     )
        #                 x_backward = Gaussian.conditional(
        #                     level_data[k]['x_prev'],
        #                     (stack ( level_data[k]['x_prior'],  level_data[k]['u_prior']) ),
        #                     ( x_backwards[k], u_backward(t-1) ),
        #                     ( stack (level_data[k]['Px_prev_x_prior'], level_data[k]['Px_prev_x_prior'],
        #                 )
        #
        #                 y_post = level_data[k-1]['u_post']
        #
        #
        #
        #         # x(T-i-1|T-i-1), x(T-i|T-i-1),  x(T-i|T-i), Cov(x(T-i-1|T-i-1), x(T-i|T-i-1))
        #         x_prev, x_prior, x_post, y_prior, Px_prev_x_prior, Pxy = trajectory[-i-1].values()
        #
        #         # x(t|T) = x(t|t) + S[x(t|t);x(t+1|t)] {S[x(t+1|t);x(t+1|t)]}^-1 ( x(t+1|T) - x(t+1|t) )
        #         # back(t) = prev(t) + cov(prev(t),prior(t+1)) cov(prior(t+1))^-1 ( back(t+1) - prior(t+1) )
        #         x_backward = Gaussian.conditional(x_prev, x_prior, x_backward, Px_prev_x_prior)
        #
        #         trajectory[-i-1]['x_backward'] = x_backward
        trajectory = [
            OrderedDict(
                **{k: [t[i][k] for t in trajectory]
                   for k in trajectory[0][0].keys()}
            )
            for i in range(self._n_systems)
        ]

        for i in range(len(trajectory)):
            for k, traj in trajectory[i].items():
                if isinstance(traj[0], Gaussian):
                    trajectory[i][k] = stack_distributions(traj, dim=1)
                elif traj[0] is None:
                    trajectory[i][k] = None
                else:
                    trajectory[i][k] = torch.stack(traj, dim=1)

        return trajectory

    def complete_data_likelihood(self, y, trajectory=None, per_timestep=True):
        if trajectory is None:
            trajectory = self.filter(y, backward_pass=True)

        x_ref = trajectory['x_backward'] if 'x_backward' in trajectory.keys() else trajectory['x_post']

        ll_x = 0
        ll_y = 0

        for i in tqdm(range(x_ref.batch_shape[1])):
            if i > 0:
                x_pred = self.fwd_transform.call(x_ref[:, i-1].mean)
                ll_x += Gaussian(x_pred, self._parameters['process_noise_cov']).log_prob(x_ref[:, i].mean)
            y_pred = self.obs_transform.call(x_ref[:, i].mean)
            ll_y += Gaussian(y_pred, self._parameters['observation_noise_cov']).log_prob(y[:, i])

        if per_timestep:
            ll_x /= (x_ref.batch_shape[1] - 1)
            ll_y /= (x_ref.batch_shape[1])

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
        x = [system['initial_state_mean'].unsqueeze(0) for system in self._systems]

        trajectory = []

        for i in range(n_points):
            level_traj = []
            for k in range(self._n_systems):
                if k == 0:
                    x_new = self._systems[k]['fwd_transform'].call(x[k])
                    y_new = self._systems[k]['obs_transform'].call(x_new)
                else:
                    x_new = self._systems[k]['fwd_transform'].call(x[k], level_traj[k-1]['y'])
                    y_new = self._systems[k]['obs_transform'].call(x_new)

                level_traj.append(dict(x=x_new, y=y_new))
            trajectory.append(level_traj)

            x = [traj['x'] for traj in level_traj]

        trajectory_ = []
        for i, system in enumerate(self._systems):
            x = Gaussian(torch.concat([t[i]['x'] for t in trajectory], dim=0),
                         system['process_noise_cov'].expand((n_points, system['state_dim'], system['state_dim'])))
            y = Gaussian(torch.concat([t[i]['y'] for t in trajectory], dim=0),
                         system['obs_noise_cov'].expand((n_points, system['obs_dim'], system['obs_dim'])))
            trajectory_.append(dict(x=x, y=y))

        return trajectory_

    def sample(self, n_points):
        x = [Gaussian.with_covariance_handle(
                system['initial_state_mean'],
                system['initial_state_cov']).sample().unsqueeze(0)
            for system in self._systems]

        process_noises = [
            Gaussian.with_covariance_handle(torch.zeros(system['state_dim']),
                     system['process_noise_cov'])
            for system in self._systems
        ]

        observation_noises = [
            Gaussian.with_covariance_handle(torch.zeros(system['obs_dim']),
                     system['obs_noise_cov'])
            for system in self._systems
        ]

        trajectory = []

        for i in range(n_points):
            level_traj = []
            for k in range(self._n_systems):
                if k == 0:
                    x_new = self._systems[k]['fwd_transform'].call(x[k]) + process_noises[k].sample((1,))
                    y_new = self._systems[k]['obs_transform'].call(x_new) + observation_noises[k].sample((1,))
                else:
                    x_new = self._systems[k]['fwd_transform'].call(x[k], level_traj[k-1]['y']) + process_noises[k].sample((1,))
                    y_new = self._systems[k]['obs_transform'].call(x_new) + observation_noises[k].sample((1,))

                level_traj.append(dict(x=x_new, y=y_new))
            trajectory.append(level_traj)

            x = [traj['x'] for traj in level_traj]

        trajectory_ = []
        for i, system in enumerate(self._systems):
            x = torch.concat([t[i]['x'] for t in trajectory], dim=0)
            y = torch.concat([t[i]['y'] for t in trajectory], dim=0)
            trajectory_.append(dict(x=x, y=y))

        return trajectory_