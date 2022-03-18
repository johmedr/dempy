import torch.distributions as td
from torch.distributions import constraints
import abc
import torch
import numpy as np


class Gaussian(td.MultivariateNormal):
    def __add__(self, o: td.MultivariateNormal):
        if self.event_shape != o.event_shape:
            raise ValueError("Event shapes must match")

        if self.batch_shape != o.batch_shape:
            raise ValueError("Batch shapes must match")

        return Gaussian(self.mean + o.mean, self.covariance_matrix + o.covariance_matrix)

    def __radd__(self, o: td.MultivariateNormal):
        return self.__add__(o)

    def __sub__(self, o: td.MultivariateNormal):
        if self.event_shape != o.event_shape:
            raise ValueError("Event shapes must match")

        if self.batch_shape != o.batch_shape:
            raise ValueError("Batch shapes must match")

        return Gaussian(self.mean - o.mean, self.covariance_matrix + o.covariance_matrix)

    def __rsub__(self, o: td.MultivariateNormal):
        return self.__sub__(o)

    @staticmethod
    def conditional(x_prior: td.MultivariateNormal, y_prior: td.MultivariateNormal, y: torch.Tensor, Pxy: torch.Tensor):
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

        K = torch.bmm(Pxy,  y_prior.covariance_matrix.inverse())
        mx_post = x_prior.mean + torch.bmm(K, (y - y_prior.mean).unsqueeze(-1)).squeeze(-1)
        Px_post = x_prior.covariance_matrix - torch.bmm(torch.bmm(K, y_prior.covariance_matrix), K.swapaxes(1,2))

        return Gaussian(mx_post, Px_post)

class GaussianTransform(abc.ABC, td.Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    @abc.abstractmethod
    def apply(self, x: td.MultivariateNormal, full: bool):
        pass

    def _call(self, x):
        return self.apply(x, full=False)


class UnscentedTransform(GaussianTransform):
    """https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf"""
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, func):
        super().__init__()
        self._func = func

    def apply(self, x: td.MultivariateNormal, full=False):
        """
        When full is true, computes the input/output covariance
        """
        mx = x.mean
        n = x.event_shape[0]

        # Compute sigma points
        kappa = max(0, n - 3)
        U = torch.linalg.cholesky((n + kappa) * x.covariance_matrix, upper=True)

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

        if full:
            # Compute input/output covariance
            res_x = (sigmas - mx.unsqueeze(0))
            Pxy = torch.einsum('i,ijk,ijl->jkl', weights, res_x, res_y)

            return td.MultivariateNormal(my, Py), Pxy
        else:
            return td.MultivariateNormal(my, Py)


class LinearizedTransform(GaussianTransform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, func, create_graph=False):
        super().__init__()
        self._func = func
        self._J = lambda x: torch.autograd.functional.jacobian(func, x, create_graph=create_graph)

    def apply(self, x: td.MultivariateNormal, full=False):
        my = self._func(x.mean)
        J = torch.stack([self._J(x.mean[i]) for i in range(x.batch_shape[0])], dim=0)
        Pxy = torch.bmm(x.covariance_matrix, J.swapaxes(1, 2))
        Py = torch.bmm(J, Pxy)

        if full:
            return td.MultivariateNormal(my, Py), Pxy
        else:
            return td.MultivariateNormal(my, Py)


class LinearTransform(GaussianTransform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, A, b=None):
        super().__init__()
        self._A = A
        if b is None:
            b = torch.zeros((A.shape[0],))
        self._b = b

    def apply(self, x: td.MultivariateNormal, full=False):
        A, b = self._A.expand((*x.batch_shape, *self._A.shape)), self._b.expand((*x.batch_shape, *self._b.shape))
        my = torch.bmm(A, x.mean.unsqueeze(-1)).squeeze(-1) + b
        Py = torch.bmm(torch.bmm(A, x.covariance_matrix), A.swapaxes(1, 2))

        if full:
            Pxy = torch.bmm(x.covariance_matrix, A.swapaxes(1, 2))
            return td.MultivariateNormal(my, Py), Pxy
        else:
            return td.MultivariateNormal(my, Py)

