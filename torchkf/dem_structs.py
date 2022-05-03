import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def block_matrix(nested_lists): 
    # a is a list of list [[[], [], tensor], [[], tensor, []]]
    # row and column size must be similar
    sizes = np.zeros((len(nested_lists), len(nested_lists[0]), 2))
    for i, row in enumerate(nested_lists): 
        for j, e in enumerate(row): 
            sizes[i, j, :] = e.shape if len(e) > 0 else (0, 0)
    row_sizes = sizes.max(1)[:, 0].astype(int)
    col_sizes = sizes.max(0)[:, 1].astype(int)
    
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