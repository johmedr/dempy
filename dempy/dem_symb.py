import numpy as np
import symengine as si

from itertools import chain, starmap, product

from .dem_structs import *
from .dem_defaults import dem_defaults
from .utils import prod

def wrap_xvp(f): 
    def _wraps(x,v,p): 
        return np.array(
            f(np.fromiter(chain(x.flat,v.flat,p.flat), dtype='d')), dtype='d')
    return _wraps


def compile_symb_func(func, *dims, input_keys=None):
    if input_keys is None: 
        import string
        input_keys = string.ascii_lowercase[:len(dims)]
    else: 
        assert(len(dims) == len(input_keys))

    dims = [(dim,1) if isinstance(dim, int) else dim for dim in dims]
    flatdims = [prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', si.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    args = [v[1] for v in var]
    symret = func(*args)

    unpackvars = [*chain(*map(lambda v: v[1].flat, symvars))]

    func = wrap_xvp(si.lambdify(unpackvars, symret, **dem_defaults.symengine.lambdify))
    func = lambda x, v, p, _shape=(symret.shape[0], 1), _func=func: _func(x,v,p).reshape(_shape)

    return func

def compile_symb_func_delays(func, *dims, delays=None, delays_idxs=None, input_keys=None):
    if input_keys is None: 
        import string
        input_keys = string.ascii_lowercase[:len(dims)]
    else: 
        assert(len(dims) == len(input_keys))

    dims = [(dim,1) if isinstance(dim, int) else dim for dim in dims]
    flatdims = [prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', si.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    args = [v[1] for v in var]
    
    f_sym = func(*args)

    f_sym = si.Matrix(f_sym.tolist())
    J = f_sym.jacobian(si.Matrix(args[0].tolist()))

    if callable(delays):
        # unpack the delay parameters
        delay_arg = var[2][1][delays_idxs]
        delay_arg = delay_arg.reshape(delay_arg.shape)

        # compute delay jacobian  
        T = delays(delay_arg)
    else: 
        T = delays

    Q = (np.eye(*T.shape) + T * J)
    Q = si.Matrix(Q.tolist()).inv() # Q is 2d (nx, nx)
    
    symret = Q @ f_sym

    unpackvars = [*chain(*map(lambda v: v[1].flat, symvars))]

    func = wrap_xvp(si.lambdify(unpackvars, symret, **dem_defaults.symengine.lambdify))
    func = lambda x, v, p, _shape=(symret.shape[0], 1), _func=func: _func(x,v,p).reshape(_shape)

    return func


def compute_sym_df_d2f(func, *dims, input_keys=None, wrt=None):
    """ 
    Use symbolic differentiation to compute jacobian and hessian of a function of 3 vectors. 
     - func: if the function to differentiate (must return a vector, ie a tensor (l, ...) where ... are empty or 1's)
     - dims: a list of tuple containing the dimensions of each argument
    Returns: (df, d2f) where: 
     - df.dx, df.dv, and df.dp contains the jacobians wrt each argument
     - d2f.dx.dx, ... contains the ndim-hessians wrt each pair of arguments
    Note that for performance, only d2f[ki][kj] with input_keys.index(ki) < input_keys.index(kj) is populated
    while d2f[kj][ki] is not. 
    Use that fact that 'd2f[kj][ki] = d2f[ki][kj].swapaxes(1, 2)' to get it.  
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
    flatdims = [prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', si.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    # arguments for calling the function (numpy.ndarrays column vectors)
    args = [v[1] for v in var]

    # Call function 
    # -------------
    fxvp = si.Matrix(func(*args).tolist())
    l = fxvp.shape[0]

    # arguments to lambdify wrt 
    unpackvars = [*chain(*map(lambda v: v[1].flat, symvars))]

    # arguments to differentiate wrt
    symvars = [*map(lambda v: (v[0], si.Matrix(v[1].tolist())), symvars)]

    # Compute first-order derivatives 
    # -------------------------------
    dfsymb  = dotdict({
        d: si.Matrix(fxvp.jacobian(sym))
        for d, sym in symvars
        if d in wrt
    })

    # callable dotdicts for output 
    df  = cdotdict()
    d2f = cdotdict()

    for i, (d1, sym1) in enumerate(symvars):
        if d1 not in wrt: continue 
            
        if d1 not in d2f.keys():
            d2f[d1] = cdotdict()
            
        for j, (d2, sym2) in enumerate(symvars): 
            # Compute second-order derivatives
            # --------------------------------
            
            # we only populate the upper triangle (just use .swapaxes(1, 2) to get the other side)
            if j < i: continue 

            if d2 not in wrt: continue                 
            if d2 not in d2f.keys(): 
                d2f[d2] = cdotdict()

            # use symmetry for d2f{j}/dx{i}^2 (removes n(n-1)/2 operations)
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

                ret[infer_to] = ret[infer_from]

            # general case
            else: 
                ret = np.asarray([*starmap(lambda x, v: 
                        si.diff(x, v) if x.free_symbols else x, 
                        product(dfsymb[d1], sym2))])

            # make a SymEngine matrix
            h  = si.Matrix(ret.reshape(l, sym1.shape[0]*sym2.shape[0]).tolist())

            # create a function if h has free (dependent) symbols
            if len(h.free_symbols) > 0: 
                func_h  = wrap_xvp(si.lambdify(unpackvars, h, **dem_defaults.symengine.lambdify))

                d2f[d1][d2] = lambda *_args, _func=func_h, _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    _func(*_args).reshape(_target_shape)

            else:
                d2f[d1][d2] = lambda *_args, _symb=cast(h.tolist()).reshape((l, *squeezedims[i], *squeezedims[j])): _symb
              
        # create a jacobian if J has free (dependent) symbols
        J = dfsymb[d1]
        if len(J.free_symbols) > 0:
            func_J  = wrap_xvp(si.lambdify(unpackvars, J, **dem_defaults.symengine.lambdify))

            df[d1] = lambda *_args, _func=func_J, _target_shape=(l, *squeezedims[i]): _func(*_args).reshape(_target_shape)
        else: 
            df[d1] = lambda *_args, _symb=cast(J.tolist()).reshape((l, *squeezedims[i])): _symb
        
    return df, d2f


def compute_sym_df_d2f_delays(func, *dims, delays=None, delays_idxs=None, input_keys=None, wrt=None):
    """ 
    !!! for now we assume that dims == nx,nv,np, 

    Use symbolic differentiation to compute jacobian and hessian of a function 
     - func: if the function to differentiate (must return a vector, ie a tensor (l, ...) where ... are empty or 1's)
     - dims: a list of tuple containing the dimensions of each argument
    Returns: (df, d2f) where: 
     - df.dx, df.dv, and df.dp contains the jacobians wrt each argument
     - d2f.dx.dx, ... contains the ndim-hessians wrt each pair of arguments
    Note that for performance, only d2f[ki][kj] with input_keys.index(ki) < input_keys.index(kj) is populated
    while d2f[kj][ki] is not. 
    Use that fact that 'd2f[kj][ki] = d2f[ki][kj].swapaxes(1, 2)' to get it.  
    """
    assert(delays is not None)

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
    flatdims = [prod(dim) for dim in dims]

    # create symbolic variables
    symvars = [
        (f'd{k}', si.symarray(k, dim))
        for k, dim in zip(input_keys, flatdims)
    ]

    # symbols in column numpy arrays
    var = [
        (k, symvar.reshape((-1, 1)))
        for k, symvar in symvars
    ]    
    
    # arguments for calling the function (numpy.ndarrays column vectors)
    args = [v[1] for v in var]

    # Call function 
    # -------------
    f    = func(*args)
    fxvp = si.Matrix(f.tolist())
    l = fxvp.shape[0]

    # arguments to lambdify wrt 
    unpackvars = [*chain(*map(lambda v: v[1].flat, symvars))]

    # arguments to differentiate wrt
    symvars = [*map(lambda v: (v[0], si.Matrix(v[1].tolist())), symvars)]

    # Compute first-order derivatives 
    # -------------------------------
    dfsymb  = dotdict({
        d: si.Matrix(fxvp.jacobian(sym))
        for d, sym in symvars
        if d in wrt
    })

    # Compute delay term 
    # ------------------
    if callable(delays): 
        is_fn_delay = True

        # unpack the delay parameters
        delay_arg = var[2][1][delays_idxs]
        delay_arg = delay_arg.reshape(delay_arg.shape)

        # compute delay jacobian  
        T = delays(delay_arg)
        Q = (np.eye(*T.shape) + T * dfsymb['dx'])
        Q = si.Matrix(Q.tolist()).inv() # Q is 2d (nx, nx)


        dQdT    = [si.diff(Q, vi) for vi in delay_arg.flat]
        d2QdTdT = np.stack([
                np.stack([
                    np.asarray(si.diff(dq, vi)) @ f # row: (dq*f)_i
                    for dq in dQdT
                ], axis=-1) # second-to-last columns: dvj
                for vi in delay_arg.flat
            ], axis=-1 # last column: dvk
        )

        dQdT    = np.stack(dQdT, axis=-1) # is 3d (nx, nx, nt)
        dQdT_fx = (np.swapaxes(dQdT, 1, 2) @ f.reshape((-1, 1))).squeeze(-1)

        Q = np.asarray(Q)

    else: 
        is_fn_delay = False

        T = delays
        Q = (np.eye(*T.shape) + T * dfsymb['dx'])
        Q = si.Matrix(Q.tolist()).inv()
        Q = np.asarray(Q)

    # callable dotdicts for output 
    df  = cdotdict()
    d2f = cdotdict()

    for i, (d1, sym1) in enumerate(symvars):
        if d1 not in wrt: continue 
            
        if d1 not in d2f.keys():
            d2f[d1] = cdotdict()
            
        for j, (d2, sym2) in enumerate(symvars): 
            # Compute second-order derivatives
            # --------------------------------
            
            # we only populate the upper triangle (just use .swapaxes(1, 2) to get the other side)
            if j < i: continue 

            if d2 not in wrt: continue                 
            if d2 not in d2f.keys(): 
                d2f[d2] = cdotdict()

            # use symmetry for d2f{j}/dx{i}^2 (removes n(n-1)/2 operations)
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

                ret[infer_to] = ret[infer_from]

            # general case
            else: 
                ret = np.asarray([*starmap(lambda x, v: 
                        si.diff(x, v) if x.free_symbols else x, 
                        product(dfsymb[d1], sym2))])

            H = ret.reshape((l, sym1.shape[0], sym2.shape[0]))

            # Apply delay in the general case, and replace nongeneral terms afterwise

            # warning dot with more than 2d applies as x@y = sum_k(x[..., k] * y[..., k, :])
            # (nx, ni, nj) = ( (nx, nx) @ (nx, ni, nj).sw(0, 1) ).sw(0, 1)
            H = ( Q @ H.swapaxes(0, 1) ).swapaxes(0, 1)

            # apply delays 
            if is_fn_delay: 
                if d2 == 'p' or j == 2: 
                    if d1 == 'p' or i == 2:
                        # H on between and nondelay components is dQdT @ J
                        H[:, delays_idxs, :] = np.swapaxes(dQdT, 1, 2) @ dfsymb[d1]
                        H[:, :, delays_idxs] = np.swapaxes(H[:, delays_idxs, :], 1, 2)

                        # H on the delay components T is d2QdTdT
                        H[:, delays_idxs][:, :, delays_idxs] = d2QdTdT
                    else: 
                        # H on between and nondelay components is dQdT @ J - use the fact that j >= i to compute 1 side only
                        # (nx, ni, nj) = ( (nx, nx, nj).sw(1, 2) @ (nx, ni) ).sw(1, 2)
                        H[:, :, delays_idxs] = (np.swapaxes(dQdT, 1, 2) @ dfsymb[d1]).swapaxes(1, 2)

            # make a SymEngine matrix
            H  = si.Matrix(H.reshape(l, sym1.shape[0]*sym2.shape[0]).tolist())

            # create a function if h has free (dependent) symbols
            if len(H.free_symbols) > 0: 
                func_h  = wrap_xvp(si.lambdify(unpackvars, H, **dem_defaults.symengine.lambdify))

                d2f[d1][d2] = lambda *_args, _func=func_h, _target_shape=(l, *squeezedims[i], *squeezedims[j]):\
                    _func(*_args).reshape(_target_shape)

            else:
                d2f[d1][d2] = lambda *_args, _symb=cast(H.tolist()).reshape((l, *squeezedims[i], *squeezedims[j])): _symb
        
        # compute jacobian
        J = Q @ dfsymb[d1]
        if is_fn_delay:
            if d1 == 'p' or i == 2: 
                J[:, delays_idxs] = dQdT_fx

        J = si.Matrix(J.tolist())
        
        # create a function if J has free (dependent) symbols
        if len(J.free_symbols) > 0:
            func_J  = wrap_xvp(si.lambdify(unpackvars, J, **dem_defaults.symengine.lambdify))

            df[d1] = lambda *_args, _func=func_J, _target_shape=(l, *squeezedims[i]): _func(*_args).reshape(_target_shape)
        else: 
            df[d1] = lambda *_args, _symb=cast(J.tolist()).reshape((l, *squeezedims[i])): _symb
        
    return df, d2f
