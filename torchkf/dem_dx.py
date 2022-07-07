from typing import Tuple
import torch 
import sympy

from .dem_structs import *


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
    dx = torch.linalg.matrix_exp(J)
    return dx[1:, 0, None]


def compute_sym_df_d2f(func, n, m, p, cast_to=np.ndarray):
    """ 
    Use symbolic differentiation to compute jacobian and hessian of a function of 3 vectors. 
     - func: if the function to differentiate (must return a vector, ie a tensor (l, ...) where ... are empty or 1's)
     - n: dimension of the first argument
     - m: dimension of the second argument
     - p: dimension of the third argument
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
    
    x  = np.array(sympy.symbols(f'x0:{n}')).reshape((n, 1))
    v  = np.array(sympy.symbols(f'v0:{m}')).reshape((m, 1))
    p  = np.array(sympy.symbols(f'p0:{p}')).reshape((p, 1))

    fxvp = sympy.Matrix(func(x, v, p))
    l = fxvp.shape[0]
    
    var  = [('dx', x), ('dv', v), ('dp', p)]

    dfsymb  = dotdict({
        d: fxvp.jacobian(sym)
        for d, sym in var
    })

    df  = cdotdict()
    d2f = cdotdict()

    for i, (d1, sym1) in enumerate(var):
        if d1 not in d2f.keys():
            d2f[d1] = cdotdict()
            
        for j, (d2, sym2) in enumerate(var): 
            if j < i: continue
            if d2 not in d2f.keys(): 
                d2f[d2] = cdotdict()

            h  = sympy.MutableDenseNDimArray((dfsymb[d1].reshape(l * sym1.shape[0], 1).jacobian(sym2)))
            h  = h.reshape(l, sym1.shape[0], sym2.shape[0])
            ht = sympy.permutedims(h, (0, 2, 1))
            
            if len(h.free_symbols) > 0: 
                d2f[d1][d2] = lambda x_, v_, p_, symb_=h : cast(sympy.lambdify((x, v, p), symb_, 'numpy')(x_, v_, p_))
                d2f[d2][d1] = lambda x_, v_, p_, symb_=ht: cast(sympy.lambdify((x, v, p), symb_, 'numpy')(x_, v_, p_))
            else:
                d2f[d1][d2] = cast(h)
                d2f[d2][d1] = cast(ht)
                
        if len(dfsymb[d1].free_symbols) > 0:
            df[d1] = lambda x_, v_, p_, symb_=dfsymb[d1]: cast(sympy.lambdify((x, v, p), symb_, 'numpy')(x_, v_, p_))
        else: 
            df[d1] = cast(dfsymb[d1])
        
    return df, d2f
    