"""
original project = "bgflow" https://github.com/noegroup/bgflow
copyright = MIT License
author = Jonas Köhler, Andreas Krämer, Manuel Dibak, Leon Klein, Frank Noé
"""

import torch
import numpy as np

def is_list_or_tuple(x):
    return isinstance(x, list) or isinstance(x, tuple)

def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x