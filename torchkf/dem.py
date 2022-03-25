import torch
from .transformations import *
from typing import Dict, Optional, List

class DEMInversion:
    def __init__(self, f_transforms: List[GaussianTransform], g_transforms: List[GaussianTransform]):
