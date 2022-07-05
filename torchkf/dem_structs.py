import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def block_matrix(nested_lists): 
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
                    if e.shape[0] != row_sizes[i] : 
                        raise ValueError(f'Unable to build block matrix: the number of rows at block-index ({i},{j}) (shape {e.shape}) '
                                         f'does not match that of the previous block (expected {row_sizes[i]}).')
                
                else: 
                    row_sizes[i]  = e.shape[0]

                # check for width
                if col_sizes[j] > 0:
                    if e.shape[1] != col_sizes[j]: 
                        raise ValueError(f'Unable to build block matrix: the number of columns at block-index ({i},{j}) (shape {e.shape}) '
                                         f'does not match that of the previous block (expected {col_sizes[j]}).')
                
                else: 
                    col_sizes[j] = e.shape[1]

            # check for empty columns
            # if i == len(nested_lists) - 1 and col_sizes[j] == 0:
            #     raise ValueError(f'Column {i+1} contains only empty matrices!')

        # check for empty rows
        # if row_sizes[i] == 0:
        #     raise ValueError(f'Row {i+1} contains only empty matrices!')
    
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