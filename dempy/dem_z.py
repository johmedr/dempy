import numpy as np 
import scipy as sp
import scipy.linalg
import warnings

from .dem_structs import *
from .dem_hgm import *
from .dem_dx import *
from .special_matrices import toeplitz

def dem_z(M: HierarchicalGaussianModel, N: int): 
    """ Generates states noise w and causes noise z """

    nl = len(M)
    t  = np.arange(N, dtype=np.float64)
    z  = []
    w  = []

    for i in range(nl): 
        if M[i].sv > np.exp(-16):
            # temporal correlation matrix with unit variance
            Kv = toeplitz(np.exp(-t**2 / (2 * M[i].sv**2)))
            Dv = 1. / np.sqrt(np.einsum('ij,ij->i', Kv, Kv))
            Kv = np.einsum('ij,i->ij', Kv, Dv)

        if M[i].sw > np.exp(-16):
            Kw = toeplitz(np.exp(-t**2 / (2 * M[i].sw**2)))
            Dw = 1. / np.sqrt(np.einsum('ij,ij->i', Kw, Kw))
            Kw = np.einsum('ij,i->ij', Kw, Dw)

        # prior expectation on causes
        P  = M[i].V

        # plus prior expectations
        try:
            for j in range(len(M[i].Q)):
                P = P + M[i].Q[j]*np.exp(M[i].hE[j]);
        except Exception as e: 
            warnings.warn(f'cannot compute prior expectations on causes - got error: \n {e}')

        # create causes: assume i.i.d. if precision is zero
        if P.size > 0: 
            if np.linalg.norm(P, ord=1) == 0:
                if M[i].sv > np.exp(-16):
                    zi = np.random.randn(M[i].l, N) @ Kv
                else: 
                    zi = np.random.randn(M[i].l, N)
            else: 
                if M[i].sv > np.exp(-16):
                    zi = sp.linalg.sqrtm(np.linalg.inv(P)) @ np.random.randn(M[i].l, N) @ Kv
                else: 
                    zi = sp.linalg.sqrtm(np.linalg.inv(P)) @ np.random.randn(M[i].l, N)
        else: 
            zi = np.zeros((M[i].l, N))
        z.append(zi.T)

        # prior expectation on states
        P  = M[i].W

        # plus prior expectations
        try:
            for j in range(len(M[i].R)):
                P = P + M[i].R[j]*np.exp(M[i].gE[j]);
        except Exception as e: 
            warnings.warn(f'cannot compute prior expectations on states - got error: \n {e}')

        # create states: assume i.i.d. if precision is zero
        if P.size > 0: 
            if np.linalg.norm(P, ord=1) == 0:
                if M[i].sw > np.exp(-16):
                    wi = np.random.randn(M[i].n, N) @ Kw #* dt
                else: 
                    wi = np.random.randn(M[i].n, N) 
            else:
                if M[i].sw > np.exp(-16):
                    wi = sp.linalg.sqrtm(np.linalg.inv(P)) @ np.random.randn(M[i].n, N) @ Kw #* dt
                else: 
                    wi = sp.linalg.sqrtm(np.linalg.inv(P)) @ np.random.randn(M[i].n, N) #* dt

        else: 
            wi = np.zeros((M[i].n, N))

        w.append(wi.T)

    return z, w