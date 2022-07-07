import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class cdotdict(dotdict):
    """callable dot dict""" 
    def __call__(self, *args, **kwargs):
        return dotdict({
            k: v(*args, **kwargs) if callable(v) else v
            for k, v in self.items()
        })

def kron(a, b): 
    return np.tensordot(a, b, axes=0).reshape((a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]))

def block_matrix(nested_lists): 

    # TODO: use np.blocks instead...

    # a is a list of list [[[], [], tensor], [[], tensor, []], [tensor, [], []]]
    # row and column size must be similar

    row_sizes = [0] * len(nested_lists)
    col_sizes = [0] * len(nested_lists[0])

    for i, row in enumerate(nested_lists): 
        if len(row) != len(col_sizes): 
            raise ValueError(f'Invalid number of columns at row {i + 1}:'
                             f' expected {len(col_sizes)}, got {len(row)}.')

        row_sizes[i] = 0
        for j, e in enumerate(row): 
            if len(e) > 0: 
                # check for height
                if row_sizes[i]  > 0: 
                    if e.shape[0] != row_sizes[i]: 
                        raise ValueError('Unable to build block matrix: the number of rows at block-index ({},{}) (shape {}) does not match that of the previous block (expected {}).'.format(i,j,e.shape,row_sizes[i]))
                
                else: 
                    row_sizes[i]  = e.shape[0]

                # check for width
                if col_sizes[j] > 0:
                    if e.shape[1] != col_sizes[j]: 
                        raise ValueError('Unable to build block matrix: the number of columns at block-index ({},{}) (shape {}) does not match that of the previous block (expected {}).'.format(i,j,e.shape,col_sizes[j]))
                
                else: 
                    col_sizes[j] = e.shape[1]

    arr = []
    for i, row in enumerate(nested_lists): 
        arr_row = []
        for j, e in enumerate(row): 
            if len(e) > 0: 
                arr_row.append(e)
            else: 
                arr_row.append(np.zeros((row_sizes[i], col_sizes[j])))

        arr.append(np.concatenate(arr_row, axis=1))


    return np.concatenate(arr, axis=0)

class cell(list): 
    def __init__(self, m, n): 
        super().__init__([[] for j in range(n)] for i in range(m))
    
    def __getitem__(self, index): 
        if isinstance(index, tuple): 
            i, j = index
            return self[i][j]
        else: 
            return super().__getitem__(index)
    
    def __setitem__(self, index, value): 
        if isinstance(index, tuple): 
            i, j = index
            xi = self[i]
            xi[j] = value
            self[i] = xi
        else: 
            super().__setitem__(index, value)


from functools import wraps
import torch 
def vfunc(in_sizes, out_size): 
    """ wraps a function of column vectors that return a column vector
     handles conversion from torch to numpy
     shapes is a list of number of rows (int) for each vector 
    """
    def _decorator(func):
        @wraps(func)
        def _wrapped(*args): 
            cargs = []
            for i, (arg, size) in enumerate(zip(args, in_sizes)): 
                if not isinstance(arg, np.ndarray): 
                    arg = arg.numpy()
                cargs.append(arg.reshape((size, 1)))

            ret = func(*cargs)
            return torch.from_numpy(ret.reshape((out_size, 1)))
        return _wrapped
    return _decorator
