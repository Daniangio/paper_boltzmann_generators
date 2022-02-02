"""
original project = "bgflow" https://github.com/noegroup/bgflow
copyright = MIT License
author = Jonas Köhler, Andreas Krämer, Manuel Dibak, Leon Klein, Frank Noé
"""


import torch


class Sampler(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _sample_with_temperature(self, n_samples, temperature, *args, **kwargs):
        raise NotImplementedError()
        
    def _sample(self, n_samples, *args, **kwargs):
        raise NotImplementedError()
    
    def sample(self, n_samples, temperature=1.0, *args, **kwargs) -> torch.Tensor:
        if temperature != 1.0:
            return self._sample_with_temperature(n_samples, temperature, *args, **kwargs)
        else:
            return self._sample(n_samples, *args, **kwargs)
