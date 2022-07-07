from typing import Tuple
import torch 
import sympy

from .dem_structs import *
import math

import warnings
import numpy as np
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
        f = f.unsqueeze(-1)

    # if isreg we use t as a regularization parameter   
    if isreg:
        t  = np.exp(t - torch.logdet(dfdx)/f.shape[0])

    if f.shape[0] != dfdx.shape[0]: 
        raise ValueError(f'Shape mismatch: first dim of f {f.shape} must match that of df/dx {dfdx.shape}.')
    if len(f) == len(dfdx) == 0:
        return torch.Tensor([[]])

    J = torch.Tensor(block_matrix([[np.zeros((1,1)), []], [f * t, dfdx * t]]))
    dx = torch.matrix_exp(J)
    return dx[1:, 0, None]


from sympy.utilities.autowrap import autowrap
import itertools


class autowrapnd:     
    def __init__(self, expr): 
        self._func = autowrap(expr)
    
    def __call__(self, *args): 
        try: 
            return self._func(*itertools.chain.from_iterable(_.view(-1) for _ in args))
        except:
            return self._func(*itertools.chain.from_iterable(np.array(_).flat for _ in args))

def compute_sym_df_d2f(func, *dims, input_keys=None, wrt=None, cast_to=np.ndarray):
    """ 
    Use symbolic differentiation to compute jacobian and hessian of a function of 3 vectors. 
     - func: if the function to differentiate (must return a vector, ie a tensor (l, ...) where ... are empty or 1's)
     - dims: a list of tuple containing the dimensions of each argument
    Returns: (df, d2f) where: 
     - df.dx, df.dv, and df.dp contains the jacobians wrt each argument
     - d2f.dx.dx, ... contains the ndim-hessians wrt each pair of arguments
    """
    if cast_to == np.ndarray: 
        cast = lambda x: np.array(x, dtype=np.float64)
    elif cast_to == torch.tensor: 
        cast = lambda x: torch.from_numpy(np.array(x, dtype=np.float64))
    elif callable(cast_to):
        cast = cast_to
    else: raise NotImplementedError()
        
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
        (f'd{k}', sympy.MatrixSymbol(k, *dim))
        for k, dim in zip(input_keys, dims)
    ]

    # wrap the symbols in numpy arrays
    var = [
        (k, np.array(symvar))
        for k, symvar in symvars
    ]
    
    # create flat numpy variables (for differentiation)
    flatvar = [
        (k, v.reshape((flatdim,)))
        for (k, v), flatdim in zip(var, flatdims)
    ]
    
    
    symargs = [v[1] for v in symvars]
    args = [v[1] for v in var]


    fxvp = sympy.simplify(sympy.Matrix(func(*args)))
    l = fxvp.shape[0]

    dfsymb  = dotdict({
        d: sympy.simplify(fxvp.jacobian(sym))
        for d, sym in flatvar
        if d in wrt
    })

    df  = cdotdict()
    d2f = cdotdict()

    for i, (d1, sym1) in enumerate(flatvar):
        if d1 not in wrt: continue 
            
        if d1 not in d2f.keys():
            d2f[d1] = cdotdict()
            
        for j, (d2, sym2) in enumerate(flatvar): 
            if j < i: continue
            if d2 not in wrt: continue 
                
            if d2 not in d2f.keys(): 
                d2f[d2] = cdotdict()

            h  = sympy.MutableDenseNDimArray((dfsymb[d1].reshape(l * sym1.shape[0], 1).jacobian(sym2)))
            h  = h.reshape(l, sym1.shape[0], sym2.shape[0])
            ht = sympy.permutedims(h, (0, 2, 1))
            
            h  = sympy.simplify(sympy.Matrix(h.reshape(l, sym1.shape[0]*sym2.shape[0])))
            ht = sympy.simplify(sympy.Matrix(h.reshape(l, sym1.shape[0]*sym2.shape[0])))
            
            if len(h.free_symbols) > 0: 

                d2f[d1][d2] = lambda *_args, _func=autowrap(h, args=symargs), _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    cast(_func(*_args)).reshape(_target_shape)
                d2f[d2][d1] = lambda *_args, _func=autowrap(ht, args=symargs), _target_shape=(l, *squeezedims[j], *squeezedims[i]):\
                    cast(_func(*_args)).reshape(_target_shape)
            else:
                d2f[d1][d2] = lambda *_args, _symb=h, _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    cast(_symb).reshape(_target_shape)
                d2f[d2][d1] = lambda *_args, _symb=ht, _target_shape=(l, *squeezedims[j], *squeezedims[i]):\
                    cast(_symb).reshape(_target_shape)
                
        J = dfsymb[d1]
        if len(J.free_symbols) > 0:
            df[d1] = lambda *_args, _func=autowrap(J, args=symargs), _target_shape=(l, *squeezedims[i]): \
                cast(_func(*_args)).reshape(_target_shape)
        else: 
            df[d1] = lambda *_args, _symb=J, _target_shape=(l, *squeezedims[i]): \
                cast(_symb).reshape(_target_shape)
        
    return df, d2f
    