from sympy.utilities.autowrap import autowrap
import functools
import sympy
import math
import numpy as np
import symengine as si

from itertools import chain, starmap, product, combinations_with_replacement
import math
from tqdm.autonotebook import tqdm

from .dem_structs import *

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
def wrap_xvp(f): 
    def _wraps(x,v,p): 
        return np.asarray(f(np.fromiter(chain(x.flat,v.flat,p.flat), dtype=np.float64)))
    return _wraps

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
        (f'd{k}', si.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]
    

    # create symbolic variables
    sympyvars = [
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
    unpackvars = [*chain(*map(lambda v: v[1].flat, symvars))]


    symvars = [*map(lambda v: (v[0], si.Matrix(v[1].tolist())), symvars)]
    _vars = [v[1] for v in symvars]
    
    # replace_dict = dict(chain(*starmap(zip, zip(_vars, symargs))))
    
    fxvp = si.Matrix(func(*args).tolist())
    l = fxvp.shape[0]

    
    dfsymb  = dotdict({
        d: si.Matrix(fxvp.jacobian(sym))
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
                        si.diff(*xv) if xv[0].free_symbols and ok else xv[0], 
                        zip(compute, product(dfsymb[d1], sym2)))])
                # ret = np.asarray([*starmap(lambda ok, xv: 
                #         ray.remote(si.diff).remote(*xv) if xv[0].free_symbols and ok else xv[0], 
                #         zip(compute, product(dfsymb[d1], sym2)))])
                # future = [*map(lambda e: isinstance(e, ray.ObjectRef), ret)]
                # ret[future] = ray.get(ret[future].tolist())

                ret[infer_to] = ret[infer_from]

            else: 
                ret = np.asarray([*starmap(lambda x, v: 
                        si.diff(x, v) if x.free_symbols else x, 
                        product(dfsymb[d1], sym2))])
                
                # ret = np.asarray([*starmap(lambda x, v: 
                #         ray.remote(si.diff).remote(x,v) if x.free_symbols else x, 
                #         product(dfsymb[d1], sym2))])

            
                # future = [*map(lambda e: isinstance(e, ray.ObjectRef), ret)]
                # ret[future] = ray.get(ret[future].tolist())
            
            # h = ret.tolist()

            
#             h  = (dfsymb[d1].reshape(l * sym1.shape[0], 1).jacobian(sym2))
            # h  = si.MutableDenseNDimArray(h)
            h  = ret.reshape(l, sym1.shape[0], sym2.shape[0])
            h  = si.Matrix(h.reshape(l, sym1.shape[0]*sym2.shape[0]).tolist())

            if len(h.free_symbols) > 0: 
                # h       = h.xreplace(replace_dict)
                func_h  = wrap_xvp(si.lambdify(unpackvars, h, cse=True))

                d2f[d1][d2] = lambda *_args, _func=func_h, _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    _func(*_args).reshape(_target_shape)

                d2f[d2][d1] = lambda *_args, _func=func_h, _interm_shape=(l, sym1.shape[0], sym2.shape[0]), _target_shape=(l, *squeezedims[j], *squeezedims[i]):\
                    _func(*_args).reshape(_interm_shape).swapaxes(1, 2).reshape(_target_shape)
            else:
                d2f[d1][d2] = lambda *_args, _symb=cast(h.tolist()).reshape((l, *squeezedims[i], *squeezedims[j])): _symb
                d2f[d2][d1] = lambda *_args, _symb=cast(h.tolist()).reshape((l, sym1.shape[0], sym2.shape[0])).swapaxes(1, 2).reshape((l, *squeezedims[j], *squeezedims[i])): _symb
              
        J = si.Matrix(dfsymb[d1].tolist())
        if len(J.free_symbols) > 0:
            # J = J.xreplace(replace_dict)
            func_J  = wrap_xvp(si.lambdify(unpackvars, J, cse=True))

            df[d1] = lambda *_args, _func=func_J, _target_shape=(l, *squeezedims[i]): _func(*_args).reshape(_target_shape)
        else: 
            df[d1] = lambda *_args, _symb=cast(J.tolist()).reshape((l, *squeezedims[i])): _symb
        
    return df, d2f
