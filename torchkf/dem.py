import torch
from .transformations import *
from typing import Dict, Optional, List, Callable
import abc
import numpy as np


class GaussianGenerativeModel: 
    def __init__(self, f, g, m, n, l, pE, pC, hE, hC, gE, gC, Q, R, V, W, xP): 
        self.f  : Callable     = f          # forward function
        self.g  : Callable     = g          # observation function

        self.m  : int          = m          # number of inputs
        self.n  : int          = n          # number of states
        self.l  : int          = l          # number of outputs

        self.pE : torch.Tensor = pE         # prior expectation of parameters p
        self.pC : torch.Tensor = pC         # prior covariance of parameters p
        self.hE : torch.Tensor = hE         # prior expectation of hyperparameters h (log-precision of cause noise)
        self.hC : torch.Tensor = hC         # prior covariance of hyperparameters h (log-precision of cause noise)
        self.gE : torch.Tensor = gE         # prior expectation of hyperparameters g (log-precision of state noise)
        self.gC : torch.Tensor = gC         # prior covariance of hyperparameters g (log-precision of state noise)

        self.Q  : torch.Tensor = Q          # precision components (input noise)
        self.R  : torch.Tensor = R          # precision components (state noise)
        self.V  : torch.Tensor = V          # fixed precision (input noise)
        self.W  : torch.Tensor = W          # fixed precision (state noise)
        self.xP : torch.Tensor = xP         # precision (states)

        self.sv : torch.Tensor = sv         # smoothness (input noise)
        self.sw : torch.Tensor = sw         # smoothness (state noise)




class DEMInversion: 
    def __init__(self, systems: List[GaussianGenerativeModel], states_embedding_order: int, causes_embedding_order: int): 
        DEMInversion.check_systems(systems)

        self.M  : List = systems                                    # systems, from upper-most (index 0) to lower-most (index -1)
        self.n  : int  = states_embedding_order                     # embedding order of states
        self.d  : int  = causes_embedding_order                     # embedding order of causes
        self.nl : int  = len(systems)                               # number of levels
        self.nv : int  = sum(M.m for M in self.M)                   # number of v (causal states)
        self.nx : int  = sum(M.n for M in self.M)                   # number of x (hidden states)
        self.ny : int  = self.M[-1].l                               # number of y (model output)
        self.nc : int  = self.M[0].l                                # number of c (prior causes)
        self.nu : int  = self.d * self.nv + self.n * self.nx        # number of generalized states

    @staticmethod
    def check_systems(systems):
        """ Checks for output/input mismatch """ 
        if len(systems) < 2:
            return

        for k in range(1, len(systems)):
            if not systems[k-1].l == systems[k].m:
                raise ValueError(f"Dimension mismatch between index {k}.obs_dim "
                                 f"({systems[k-1].l}) and index {k+1}.input_dim ({systems[k-1].m}) "
                                 f"(indexing from 1)")



    def run(self, y, nD=1, nE=8, nM=8, K=1, tol=np.exp(-4)): 
        M  : List = self.M
        n  : int  = self.n
        d  : int  = self.d
        nl : int  = self.nl 
        nv : int  = self.nv
        nx : int  = self.nx
        ny : int  = self.nc
        nu : int  = self.nu 
        nT : int  = y.shape[1]

        for iE in range(nE): 
            for iT in range(nT): 
                update_x()
            for iM in range(nM): 
                update_h()
            fa[iT] = compute_fa()
            if fa[iT] > fa[iT - 1]: 
                update_p()
            uiC = dEduu
            piC = P +ddpp
            hiC = -dUdhh


 


class DEMHierarchicalDynamicalModel:
    def __init__(self, systems: List[GaussianGenerativeModel], order=4):
        DEMHierarchicalDynamicalModel.check_systems(systems)
        self._systems = systems
        self._n_systems = len(systems)
        self._input_dim = systems[0]['input_dim']
        self._obs_dim = systems[-1]['obs_dim']
        self._input_mean = input_mean
        self._input_cov = input_cov
        self._order = order

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
        return ydef predef prepare_data(self, y):
        """ y: torch tensor ([n_batchs,] n_times, obs_dim)"""
        if not y.shape[-1] == self._obs_dim:
            raise ValueError(
                f"Dimension of measurements y ({y.shape[-1]} does not match declared obs_dim ({self._obs_dim}).")
        if len(y.shape) == 2:
            y = y.unsqueeze(0)
        elif len(y.shape) != 3:
            raise ValueError(
                f"Expected measurements y to have 2 or 3 dimensions, but got ({len(y.shape)}.")
        return ypare_data(self, y):
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

    def to_generalized_coordinates(self, series): 
        """ series: torch tensor (n_batchs, n_times, dim) 
            inspired from spm_DEM_embed.m
            series_generalized = T * [series[t-order/2], dots, series[t+order/2]]
        """
        n_batchs, n_times, dim = series.shape
        p = self._order

        E = torch.zeros((p + 1, p + 1))
        times = torch.arange(n_times)

        # Create E_ij(t) (note that indices start at 0) 
        for i in range(p + 1): 
            for j in range(p + 1): 
                E[i, j] = (i + 1 - int((p + 1) / 2))**(j) / np.math.factorial(j) 

        # Compute T
        T = torch.linalg.inv(E)

        # Compute the slices
        slices = []
        for t in times: 
            start = int(t - (p + 1) / 2)
            end = start + p + 1
            if start < 0: 
                slices.append(slice(0, p + 1))
            elif end > n_times:
                slices.append(slice(n_times - (p + 1), n_times))
            else: 
                slices.append(slice(start, end))

        series_slices = torch.stack([series[:, _slice] for _slice in slices], dim=1)

        # series_slices is (n_batchs, n_times, order + 1, dim)
        # T is ( order+1, order+1)
        generalized_coordinates = torch.einsum('ijkl,mk->ijlm', series_slices, T)


        return generalized_coordinates

    def generalized_covariance(self, s):
        """ s is the roughtness of the noise process 
        """
        p = self._order
        k = torch.arange(p + 1)
        x = np.sqrt(2) * s
        r = np.cumprod(1 - 2 * k) / (x**(2*k))
        S = torch.zeros((p+1,p+1))
        for i in range(p + 1): 
            j = 2 * k - i
            filt = torch.logical_and(j >= 0, j < p + 1)
            S[i,j[filt]] = (-1) ** (i) * r[filt]
        R = torch.linalg.inv(S)
        return S, R

    def filter(self, y):
        y = self.prepare_data(y)
        y = self.generalized_coordinates(y)
        D = torch.diagonal(torch.ones(self._order), k=1)
        