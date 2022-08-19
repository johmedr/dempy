import numpy as np

from .dem_symb import *
from .utils import *


def compute_dx(f, dfdx, t, isreg=False): 
    # Adapted from spm_dx
    # Compute updates using local linearization (Ozaki 1985)

    if len(f.shape) == 1: 
        f = f[..., None]

    # if isreg we use t as a regularization parameter   
    if isreg:
        t  = np.exp(t - np.linalg.slogdet(dfdx)[1] / f.shape[0])

    if f.shape[0] != dfdx.shape[0]: 
        raise ValueError(f'Shape mismatch: first dim of f {f.shape} must match that of df/dx {dfdx.shape}.')

    if len(f) == len(dfdx) == 0:
        return np.array([[]])

    # use the exponentiation trick to avoid inverting dfdx
    J  = block_matrix([
        [np.zeros((1,1)),       []], 
        [          f * t, dfdx * t]
    ])
    dx = matrix_exp(J)

    return dx[1:, 0, None]





