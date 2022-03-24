import torch.distributions as td
from torch.distributions import constraints
import abc
import torch
import numpy as np
from typing import Union, Optional


class Gaussian(td.MultivariateNormal):
    @staticmethod
    def with_covariance_handle(loc, cov, tol=1e-6):
        try:
            return Gaussian(loc, cov)
        except ValueError:
            pass
        cov = 0.5 * (cov + cov.swapaxes(-1, -2))
        try:
            return Gaussian(loc, cov)
        except ValueError:
            pass

        cov = cov + tol * torch.eye(cov.shape[-1]).expand(cov.shape)
        return Gaussian(loc, cov)

    def __add__(self, o: td.MultivariateNormal):
        if self.event_shape != o.event_shape:
            raise ValueError("Event shapes must match")

        if self.batch_shape != o.batch_shape:
            raise ValueError("Batch shapes must match")

        return Gaussian.with_covariance_handle(self.mean + o.mean, self.covariance_matrix + o.covariance_matrix)

    def __radd__(self, o: td.MultivariateNormal):
        return self.__add__(o)

    def __sub__(self, o: td.MultivariateNormal):
        if self.event_shape != o.event_shape:
            raise ValueError("Event shapes must match")

        if self.batch_shape != o.batch_shape:
            raise ValueError("Batch shapes must match")

        return Gaussian.with_covariance_handle(self.mean - o.mean, self.covariance_matrix + o.covariance_matrix)

    def __rsub__(self, o: td.MultivariateNormal):
        return self.__sub__(o)

    def __getitem__(self, item):
        return Gaussian.with_covariance_handle(self.mean.__getitem__(item), self.covariance_matrix.__getitem__(item))

    @staticmethod
    def conditional(x_prior: td.MultivariateNormal,
                    y_prior: td.MultivariateNormal, y: Union[torch.Tensor, td.MultivariateNormal], Pxy: torch.Tensor):
        """ Computes x|y from x, y """
        # if len(y) == 1:
        #     y = y.unsqueeze(0)
        # if x_prior.batch_shape != y.shape[0]:
        #     x_prior = x_prior.expand(torch.Size(y.shape[0]))
        # if len(y_prior.batch_shape) == 0:
        #     y_prior = y_prior.expand(torch.Size(y.shape[0]))
        # elif len(y_prior.batch_shape) ==
        # if len(Pxy.shape) == 2:
        #     Pxy = Pxy.expand(y.shape[0], Pxy.shape)
        # else:

        if isinstance(y, torch.Tensor):
            y_ = y
            Py_ = y_prior.covariance_matrix
        elif isinstance(y, td.MultivariateNormal):
            y_ = y.mean
            Py_ = y_prior.covariance_matrix - y.covariance_matrix
            print(Py_, y_prior.covariance_matrix, y.covariance_matrix)
        else: raise ValueError('y must be MultivariateNormal or Tensor')

        K = torch.bmm(Pxy,  y_prior.covariance_matrix.inverse())
        mx_post = x_prior.mean + torch.bmm(K, (y_ - y_prior.mean).unsqueeze(-1)).squeeze(-1)
        Px_post = x_prior.covariance_matrix - torch.bmm(torch.bmm(K, Py_), K.swapaxes(1,2))

        return Gaussian.with_covariance_handle(mx_post, Px_post)


def stack_distributions(list_of_gaussians, dim=1):
    mean, cov = torch.stack([g.mean for g in list_of_gaussians], dim=dim), torch.stack([g.covariance_matrix for g in list_of_gaussians], dim=dim)
    return Gaussian(mean, cov)


class GaussianTransform(abc.ABC, td.Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    @abc.abstractmethod
    def apply(self, x: td.MultivariateNormal, full: bool):
        pass

    @abc.abstractmethod
    def call(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        pass

    def _call(self, x):
        return self.apply(x, full=False)


class UnscentedTransform(GaussianTransform):
    """https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf"""
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, func):
        """
        - func: should take a single input!
        """
        super().__init__()
        self._func = func

    def call(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        if u is not None:
            x = torch.hstack([x, u])
        return self._func(x)

    def apply(self, x: td.MultivariateNormal, u: Optional[td.MultivariateNormal] = None, full=False):
        """
        When full is true, computes the input/output covariance
        """
        if u is not None:
            mx = torch.hstack([x.mean, u.mean])
            Px = torch.zeros((*u.batch_shape, mx.shape[-1], mx.shape[-1]))
            Px[..., :x.event_shape[0], :x.event_shape[0]] = x.covariance_matrix
            Px[..., x.event_shape[0]:, x.event_shape[0]:] = u.covariance_matrix
        else:
            mx = x.mean
            Px = x.covariance_matrix

        n = mx.shape[-1]

        # Compute sigma points
        kappa = max(0, n - 3)
        U = torch.linalg.cholesky((n + kappa) * Px, upper=True)

        sigmas = mx.repeat(2 * n + 1, 1, 1)  # shape is (2 * n + 1, n)
        for k in range(n):
            sigmas[k + 1] += U[:, k]
            sigmas[n + k + 1] -= U[:, k]

        # Compute weights
        weights = torch.full((2 * n + 1,), 0.5 * kappa / (n + kappa))
        weights[0] = kappa / (n + kappa)

        # Pass through function
        y = torch.stack([self._func(sigmas[k]) for k in range(2 * n + 1)], dim=0)

        # Computes output moments
        my = torch.einsum('i,ijk->jk', weights, y)
        res_y = (y - my.unsqueeze(0))
        Py = torch.einsum('i,ijk,ijl->jkl', weights, res_y, res_y)
        # Py = 0.5 * (Py + Py.swapaxes(1, 2))

        if full:
            # Compute input/output covariance
            res_x = (sigmas - mx.unsqueeze(0))
            Pxy = torch.einsum('i,ijk,ijl->jkl', weights, res_x, res_y)

            return Gaussian.with_covariance_handle(my, Py), Pxy
        else:
            return Gaussian.with_covariance_handle(my, Py)


class LinearizedTransform(GaussianTransform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, func, create_graph=False):
        super().__init__()
        self._func = func
        self._J = lambda x: torch.autograd.functional.jacobian(func, x, create_graph=create_graph)

    def call(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        if u is not None:
            x = torch.hstack([x, u])
        return self._func(x)

    def apply(self, x: td.MultivariateNormal, u: Optional[td.MultivariateNormal] = None, full=False):

        if u is not None:
            mx = torch.hstack([x.mean, u.mean])
            Px = torch.zeros((*u.batch_shape, mx.shape[-1], mx.shape[-1]))
            Px[..., :x.event_shape[0], :x.event_shape[0]] = x.covariance_matrix
            Px[..., x.event_shape[0]:, x.event_shape[0]:] = u.covariance_matrix
        else:
            mx = x.mean
            Px = x.covariance_matrix

        my = self._func(mx)
        J = torch.stack([self._J(mx[i]) for i in range(x.batch_shape[0])], dim=0)
        Pxy = torch.bmm(Px, J.swapaxes(1, 2))
        Py = torch.bmm(J, Pxy)
        # Py = 0.5 * (Py + Py.swapaxes(1, 2))

        if full:
            return Gaussian.with_covariance_handle(my, Py), Pxy
        else:
            return Gaussian.with_covariance_handle(my, Py)


class LinearTransform(GaussianTransform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, A, b=None):
        super().__init__()
        self._A = A
        if b is None:
            b = torch.zeros((A.shape[0],))
        self._b = b

    def call(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        if u is not None:
            x = torch.hstack([x, u])
        A, b = self._A.expand((*x.shape[:-1], *self._A.shape)), self._b.expand((*x.shape[:-1], *self._b.shape))
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) + b

    def apply(self, x: td.MultivariateNormal, u: Optional[td.MultivariateNormal] = None, full=False):
        if u is not None:
            mx = torch.hstack([x.mean, u.mean])
            Px = torch.zeros((*u.batch_shape, mx.shape[-1], mx.shape[-1]))
            Px[..., :x.event_shape[0], :x.event_shape[0]] = x.covariance_matrix
            Px[..., x.event_shape[0]:, x.event_shape[0]:] = u.covariance_matrix
        else:
            mx = x.mean
            Px = x.covariance_matrix

        A, b = self._A.expand((*x.batch_shape, *self._A.shape)), self._b.expand((*x.batch_shape, *self._b.shape))
        my = torch.bmm(A, mx.unsqueeze(-1)).squeeze(-1) + b
        Pxy = torch.bmm(Px, A.swapaxes(1, 2))
        Py = torch.bmm(A, Pxy)
        # Py = 0.5 * (Py + Py.swapaxes(1, 2))

        if full:
            return Gaussian.with_covariance_handle(my, Py), Pxy
        else:
            return Gaussian.with_covariance_handle(my, Py)

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape((-1, self._A.shape[1]))
        y = y.reshape((-1, self._A.shape[0]))
        self._A = torch.linalg.lstsq(x, y).solution.T
        self._b = torch.mean(y - x @ self._A.T, dim=0)