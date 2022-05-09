from typing import Tuple
import torch 

from .dem_structs import *


def compute_df_d2f(func, inputs, input_keys=None) -> Tuple[dotdict, dotdict]:
    """ compute first- and second-order derivatives of `func` evaluated at `inputs`. 
    Returns a tuple of (df, d2f) where: 
    df.dk is the derivative wrt input indexed  by input key 'dk'
    d2f.di.dj is the 2nd-order derivative wrt inputs 'di' and 'dj'. 
    """
    if input_keys is None:
        input_keys = [f'dx{i}' for i in range(len(inputs))]
    def handle_shapes(inputs):
        xs = []
        for x in inputs:
            if any(_ == 0 for _ in x.shape) or len(x.shape) == 0: 
                xs.append(torch.Tensor())
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
        if any(_ > 0 for _ in Ji[i].shape):
            df[input_keys[i]] = Ji[i]
        else: 
            df[input_keys[i]] = torch.Tensor() 
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
                Hij[input_keys[j]] = torch.Tensor()
        d2f[input_keys[i]] = Hij
    # dim 1 of H are df[i]/d(x[:]x[:])
    # dim 2 of H are df[:]/d(x[i]x[:])
    # dim 3 of H are df[:]/d(x[:]x[j])
    # d2f is (Nf, Na, Nb)
    return df, d2f

def compute_dx(f, dfdu, t, isreg=False): 
    if len(f.shape) == 1: 
        f = f.unsqueeze(-1)
    if isreg:
        t  = np.exp(t - torch.logdet(dfdu)/f.shape[0]);
    J = torch.Tensor(block_matrix([[np.zeros((1,1)), []], [f * t, t * dfdu]]))
    dx = torch.linalg.matrix_exp(J)
    return dx[1:, 0, None]

