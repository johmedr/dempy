import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from numpy.core.multiarray import normalize_axis_index

def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

def hankel(r, axis=-1, dtype=None, subok=False, writeable=False):
    r = np.asarray(r, dtype)
    axis = normalize_axis_index(axis, r.ndim)
    
    H = sliding_window_view(r, (r.shape[axis] // 2 + 1,), axis=axis, subok=subok, writeable=writeable)
    
    return H

def circulant(r, axis=-1, dtype=None, subok=False, writeable=False):
    r = np.asarray(r, dtype)
    axis = normalize_axis_index(axis, r.ndim)
    
    r = np.concatenate([array_slice(r, axis, 1, r.shape[axis] + 1, 1), r], axis)
    
    C = sliding_window_view(r, (r.shape[axis] // 2 + 1,), axis=axis, subok=subok, writeable=writeable)
    C = np.flip(C, axis + 1)
    
    return C

def toeplitz(r, axis=-1, dtype=None, hermitian=True, subok=False, writeable=False): 
    r = np.asarray(r, dtype)
    axis = normalize_axis_index(axis, r.ndim)
    
    if hermitian: 
        r = np.concatenate([array_slice(r, axis, r.shape[axis],  0, -1), r], axis)
    
    T = sliding_window_view(r, (r.shape[axis] // 2 + 1,), axis=axis, subok=subok, writeable=writeable)
    T = np.flip(T, axis + 1)
    
    return T