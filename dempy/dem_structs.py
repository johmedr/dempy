class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class cdotdict(dotdict):
    """callable dot dict""" 
    def __call__(self, *args, **kwargs):
        return dotdict(
            zip(self.keys(),
                map(lambda v: 
                    v(*args, **kwargs) if callable(v) else v, 
                    self.values())))


class cell(list): 
    """emulates matlab's cell"""
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

