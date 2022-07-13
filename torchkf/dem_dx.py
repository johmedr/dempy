from typing import Tuple
import torch 
import sympy
import scipy as sp
import scipy.linalg

from .dem_structs import *
import math

import warnings
import numpy as np
import joblib

# necessary for sympy + numpy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def compute_df_d2f(func, inputs, input_keys=None) -> Tuple[dotdict, dotdict]:
    """ compute first- and second-order derivatives of `func` evaluated at `inputs`.
    inputs must be 1 or 2d, func must return 1 or 2d tensors 
    Returns a tuple of (df, d2f) where: 
    df.dk is the derivative wrt input indexed  by input key 'dk'
    d2f.di.dj is the 2nd-order derivative wrt inputs 'di' and 'dj'. 
    """
    raise NotImplementedError()
    if input_keys is None:
        input_keys = [f'dx{i}' for i in range(len(inputs))]
    else: 
        assert(len(input_keys) == len(inputs))

    def handle_shapes(inputs):
        xs     = []
        for x in inputs:
            if any(_ == 0 for _ in x.shape) or len(x.shape) == 0: 
                xs.append(torch.empty(0))
            elif len(x.shape) == 1:
                xs.append(x)
            elif x.shape[0] == x.shape[1] == 1: 
                xs.append(x.squeeze(1))
            else:
                xs.append(x.squeeze())
        return tuple(xs)
    
    inputs = handle_shapes(inputs)
    
    Ji = torch.autograd.functional.jacobian(func, inputs)
    df = dotdict()
    for i in range(len(inputs)): 
        if all(_ > 0 for _ in Ji[i].shape):
            df[input_keys[i]] = Ji[i].reshape((-1, inputs[i].shape[0]))
        else: 
            df[input_keys[i]] = torch.empty(0)
    # dim 1 of J are df[:]/dx[i]
    # dim 2 of J are df[j]/dx[:]
    
    d2f = dotdict()
    for i in range(len(inputs)): 
        # Compute d/dxj(dfdxi)
        Hi = torch.autograd.functional.jacobian(
                lambda *x: torch.autograd.functional.jacobian(func, x)[i], 
                inputs, vectorize=True)
        Hij = dotdict()
        for j in range(len(inputs)): 
            if all(_ > 0 for _ in Hi[j].shape):
                Hij[input_keys[j]] = Hi[j]
            else: 
                Hij[input_keys[j]] = torch.empty(0)
        d2f[input_keys[i]] = Hij
    # dim 1 of H are df[i]/d(x[:]x[:])
    # dim 2 of H are df[:]/d(x[i]x[:])
    # dim 3 of H are df[:]/d(x[:]x[j])
    # d2f is (Nf, Na, Nb)
    return df, d2f

def compute_dx(f, dfdx, t, isreg=False): 
    if len(f.shape) == 1: 
        f = f[..., None]

    # if isreg we use t as a regularization parameter   
    if isreg:
        t  = np.exp(t - np.linalg.slogdet(dfdx)[1]/f.shape[0])

    if f.shape[0] != dfdx.shape[0]: 
        raise ValueError(f'Shape mismatch: first dim of f {f.shape} must match that of df/dx {dfdx.shape}.')
    if len(f) == len(dfdx) == 0:
        return np.array([[]])

    J = block_matrix([[np.zeros((1,1)), []], [f * t, dfdx * t]])
    dx = torch.matrix_exp(torch.from_numpy(J)).numpy()
    return dx[1:, 0, None]





from sympy.utilities.autowrap import autowrap
import functools

from itertools import chain, starmap, product, combinations_with_replacement
import math
from tqdm.autonotebook import tqdm

import ray


@functools.lru_cache
def compile_symb_func(func, *dims, input_keys=None):
    if input_keys is None: 
        import string
        input_keys = string.ascii_lowercase[:len(dims)]
    else: 
        assert(len(dims) == len(input_keys))

    dims = [(dim,1) if isinstance(dim, int) else dim for dim in dims]
    flatdims = [math.prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', sympy.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]
    
    # create symbolic matrix symb
    symmat = [
        (f'd{k}', sympy.MatrixSymbol(k, dim, 1))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    symargs = [v[1] for v in symmat]
    args = [v[1] for v in var]
    _vars = [v[1] for v in symvars]
    
    replace_dict = dict(chain(*starmap(zip, zip(_vars, symargs))))
    
    symret = sympy.Matrix(func(*args))
    
    return sympy.lambdify(symargs, symret.xreplace(replace_dict), cse=True)


@functools.lru_cache
def compute_sym_df_d2f(func, *dims, input_keys=None, wrt=None):
    """ 
    Use symbolic differentiation to compute jacobian and hessian of a function of 3 vectors. 
     - func: if the function to differentiate (must return a vector, ie a tensor (l, ...) where ... are empty or 1's)
     - dims: a list of tuple containing the dimensions of each argument
    Returns: (df, d2f) where: 
     - df.dx, df.dv, and df.dp contains the jacobians wrt each argument
     - d2f.dx.dx, ... contains the ndim-hessians wrt each pair of arguments

    Basically, it is faster for large expressions to derive wrt symarrays, however lambdified functions perform best
    on MatrixSymbol. Thus, we differentiate wrt symarrays, .xreplace with MatrixSymbols and then lambdify
    """
    cast = lambda x: np.array(x, dtype=np.float64)
    
    if input_keys is None: 
        import string
        input_keys = string.ascii_lowercase[:len(dims)]
    else: 
        assert(len(dims) == len(input_keys))
    
    if wrt is None:
        wrt = input_keys
    else: 
        assert(all(_ in input_keys for _ in wrt))
        
    wrt = [f'd{k}' for k in wrt]
    dims = [(dim,1) if isinstance(dim, int) else dim for dim in dims]
    
    # Squeeze column vectors
    squeezedims = [dim if len(dim) == 1 or dim[1] != 1 else (dim[0],) for dim in dims]
    
    # compute flat dimension 
    flatdims = [math.prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', sympy.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]
    
    # create symbolic matrix symb
    symmat = [
        (f'd{k}', sympy.MatrixSymbol(k, dim, 1))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    symargs = [v[1] for v in symmat]
    args = [v[1] for v in var]
    _vars = [v[1] for v in symvars]
    
    replace_dict = dict(chain(*starmap(zip, zip(_vars, symargs))))
    
    fxvp = sympy.Matrix(func(*args))
    l = fxvp.shape[0]

    
    dfsymb  = dotdict({
        d: fxvp.jacobian(sym)
        for d, sym in symvars
        if d in wrt
    })

    df  = cdotdict()
    d2f = cdotdict()


    for i, (d1, sym1) in enumerate(symvars):
        if d1 not in wrt: continue 
            
        if d1 not in d2f.keys():
            d2f[d1] = cdotdict()
            
        for j, (d2, sym2) in enumerate(symvars): 
            
            if j < i: continue
            if d2 not in wrt: continue 
                
            if d2 not in d2f.keys(): 
                d2f[d2] = cdotdict()


            if i ==  j:
                compute  = [*map(lambda idxs: idxs[1] <= idxs[2], 
                    product(range(l), range(sym1.shape[0]), range(sym2.shape[0])))]
                infer_to = [*map(lambda idxs: idxs[1] > idxs[2], 
                    product(range(l), range(sym1.shape[0]), range(sym2.shape[0])))]
                infer_from = [*map(lambda idxs: idxs[1] < idxs[2], 
                    product(range(l), range(sym1.shape[0]), range(sym2.shape[0])))]

                ret = np.asarray([*starmap(lambda ok, xv: 
                        ray.remote(sympy.diff).remote(*xv) if xv[0].free_symbols and ok else xv[0], 
                        zip(compute, product(dfsymb[d1], sym2)))])
                future = [*map(lambda e: isinstance(e, ray.ObjectRef), ret)]
                ret[future] = ray.get(ret[future].tolist())

                ret[infer_to] = ret[infer_from]

            else: 
                ret = np.asarray([*starmap(lambda x, v: 
                        ray.remote(sympy.diff).remote(x,v) if x.free_symbols else x, 
                        product(dfsymb[d1], sym2))])
            
                future = [*map(lambda e: isinstance(e, ray.ObjectRef), ret)]
                ret[future] = ray.get(ret[future].tolist())
            
            h = ret.tolist()
            
#             h  = (dfsymb[d1].reshape(l * sym1.shape[0], 1).jacobian(sym2))
            h  = sympy.MutableDenseNDimArray(h)
            h  = h.reshape(l, sym1.shape[0], sym2.shape[0])
            h  = sympy.Matrix(h.reshape(l, sym1.shape[0]*sym2.shape[0]))

            if len(h.free_symbols) > 0: 
                h       = h.xreplace(replace_dict)
                func_h  = sympy.lambdify(symargs, h, cse=True)

                d2f[d1][d2] = lambda *_args, _func=func_h, _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    _func(*_args).reshape(_target_shape)

                d2f[d2][d1] = lambda *_args, _func=func_h, _interm_shape=(l, sym1.shape[0], sym2.shape[0]), _target_shape=(l, *squeezedims[j], *squeezedims[i]):\
                    _func(*_args).reshape(_interm_shape).swapaxes(1, 2).reshape(_target_shape)
            else:
                d2f[d1][d2] = lambda *_args, _symb=cast(h).reshape((l, *squeezedims[i], *squeezedims[j])): _symb
                d2f[d2][d1] = lambda *_args, _symb=cast(h).reshape((l, sym1.shape[0], sym2.shape[0])).swapaxes(1, 2).reshape((l, *squeezedims[j], *squeezedims[i])): _symb

                
        J = dfsymb[d1]
        if len(J.free_symbols) > 0:
            func_J  = sympy.lambdify(symargs, J.xreplace(replace_dict), 'numpy', cse=True)

            df[d1] = lambda *_args, _func=func_J, _target_shape=(l, *squeezedims[i]): _func(*_args).reshape(_target_shape)
        else: 
            df[d1] = lambda *_args, _symb=cast(J).reshape((l, *squeezedims[i])): _symb
        
    return df, d2f
