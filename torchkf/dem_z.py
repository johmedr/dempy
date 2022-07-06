import torch 
import numpy as np 
import scipy as sp
import scipy.linalg
from .dem_structs import *
from .dem_hgm import *
from .dem_dx import *

def dem_z(M: HierarchicalGaussianModel, N: int): 
    """ Generates states noise w and causes noise z """

    # TODO: add dt in HierarchicalGaussianModel

    nl = len(M)
    dt = M.dt
    t  = np.arange(N, dtype=np.float64) * dt
    z  = []
    w  = []

    for i in range(nl): 
        # temporal correlation matrix with unit variance
        Kv = sp.linalg.toeplitz(np.exp(-t**2 / (2 * M[i].sv**2)))
        Kv = np.diag(1. / np.sqrt( np.diag(Kv @ Kv.T) )) @ Kv
        Kv = torch.from_numpy(Kv)

        Kw = sp.linalg.toeplitz(np.exp(-t**2 / (2 * M[i].sw**2)))
        Kw = np.diag(1. / np.sqrt( np.diag(Kw @ Kw.T) )) @ Kw
        Kw = torch.from_numpy(Kw)

        # prior expectation on causes
        P  = M[i].V

        # plus prior expectations
        try:
            for j in range(len(M[i].Q)):
                P = P + M[i].Q[j]*exp(M[i].hE[j]);
        except Exception as e: 
            print(e)

        # create causes: assume i.i.d. if precision is zero
        if torch.numel(P) > 0: 
            if torch.linalg.norm(P, ord=1) == 0:
                zi = torch.randn(M[i].l, N) @ Kv
            elif torch.linalg.norm(P, ord=1) >= np.exp(16):
                zi = torch.zeros((M[i].l, N))
            else: 
                zi = torch.from_numpy(sp.linalg.sqrtm(torch.linalg.inv(P))) @ torch.randn(M[i].l, N) @ Kv
        else: 
            zi = torch.zeros((M[i].l, N))
        z.append(zi.T)

        # prior expectation on states
        P  = M[i].W

        # plus prior expectations
        try:
            for j in range(len(M[i].R)):
                P = P + M[i].R[j]*exp(M[i].gE[j]);
        except Exception as e: 
            print(e)

        # create states: assume i.i.d. if precision is zero
        if torch.numel(P) > 0: 
            if torch.linalg.norm(P, ord=1) == 0:
                wi = torch.randn(M[i].n, N) @ Kw * dt
            elif torch.linalg.norm(P, ord=1) >= np.exp(16):
                wi = torch.zeros((M[i].n, N))
            else: 
                wi = torch.from_numpy(sp.linalg.sqrtm(torch.linalg.inv(P))) @ torch.randn(M[i].n, N) @ Kw * dt
        else: 
            wi = torch.zeros((M[i].n, N))

        w.append(wi.T)

    return z, w