from .dem_structs import dotdict

import numpy as np


dem_defaults = dotdict() 

dem_defaults.symengine = dotdict()
dem_defaults.symengine.lambdify = dotdict(backend='llvm', cse=True, dtype=np.float64)