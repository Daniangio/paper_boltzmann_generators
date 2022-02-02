"""
original project = "bgflow" https://github.com/noegroup/bgflow
copyright = MIT License
author = Jonas Köhler, Andreas Krämer, Manuel Dibak, Leon Klein, Frank Noé
"""

import torch

from .base import Energy


class MultimodalEnergy(Energy):
    def __init__(self, dim):
        super().__init__(dim)

    def _energy(self, x, **kwargs):
        X1 = x[:, [0]]
        X2 = x[:, 1:]
        energy = (X1.pow(2) + X2 - 11).pow(2) + (X1 + X2.pow(2) - 7).pow(2)
        # (x^2 + y - 11)^2 + (x + y^2 -7)^2
        return (energy).view(-1)
    
    def _hessian(self, x: torch.Tensor):
        X1 = x[:, [0]]
        X2 = x[:, 1:]
        hessian = torch.zeros((len(X1), self.dim, self.dim), dtype=torch.float32).to(x.device)
        hessian[:, 0, 0] = 12 * X1.pow(2) + 4 * X2 - 42
        hessian[:, 0, 1] = (4 * (X1 + X2)).view(-1)
        hessian[:, 1, 0] = (4 * (X1 + X2)).view(-1)
        hessian[:, 1, 1] = (12 * X2.pow(2) + 4 * X1 - 26).view(-1)
        return torch.linalg.eigvals(hessian).real
