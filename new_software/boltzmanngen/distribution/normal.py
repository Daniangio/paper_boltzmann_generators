"""
original project = "bgflow" https://github.com/noegroup/bgflow
copyright = MIT License
author = Jonas Köhler, Andreas Krämer, Manuel Dibak, Leon Klein, Frank Noé
"""

import torch
import numpy as np
from .energy.base import Energy
from .sampling.base import Sampler

def _is_symmetric_matrix(m):
    return torch.allclose(m, m.t())


class NormalDistribution(Energy, Sampler):

    def __init__(self, dim, mean=None, cov=None):
        super().__init__(dim=dim)
        self._has_mean = mean is not None
        if self._has_mean:
            assert len(mean.shape) == 1, "`mean` must be a vector"
            assert mean.shape[-1] == self.dim, "`mean` must have dimension `dim`"
            self.register_buffer("_mean", mean)
        else:
            self.register_buffer("_mean", torch.zeros(self.dim))
        self._has_cov = False
        if cov is not None:
            self.set_cov(cov)
        self._compute_Z()

    def _energy(self, x):
        if self._has_mean:
            x = x - self._mean
        if self._has_cov:
            diag = torch.exp(-0.5 * self._log_diag)
            x = x @ self._rot
            x = x * diag
        return (0.5 * x.pow(2).sum(dim=-1, keepdim=True) + self._log_Z).view(-1)
    
    def _compute_Z(self):
        self._log_Z = self.dim / 2 * np.log(2 * np.pi)
        if self._has_cov:
            self._log_Z += 1 / 2 * self._log_diag.sum()
    
    @staticmethod
    def _eigen(cov):
        try:
            diag, rot = torch.linalg.eig(cov)
            assert (diag.imag.abs() < 1e-6).all(), "`cov` possesses complex valued eigenvalues"
            diag, rot = diag.real, rot.real
        except AttributeError:
            # old implementation
            diag, rot = torch.eig(cov, eigenvectors=True)
            assert (diag[:,1].abs() < 1e-6).all(), "`cov` possesses complex valued eigenvalues"
            diag = diag[:,0] 
        return diag + 1e-6, rot

    def set_cov(self, cov):
        self._has_cov = True
        assert (
            len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]
        ), "`cov` must be square matrix"
        assert (
            cov.shape[0] == self.dim and cov.shape[1] == self.dim
        ), "`cov` must have dimension `[dim, dim]`"
        assert _is_symmetric_matrix, "`cov` must be symmetric"
        diag, rot = self._eigen(cov)
        assert torch.all(diag > 0), "`cov` must be positive definite"
        self.register_buffer("_log_diag", diag.log().unsqueeze(0))
        self.register_buffer("_rot", rot)
    
    def _sample_with_temperature(self, n_samples, temperature=1.0):
        samples = torch.randn(n_samples, self.dim, dtype=self._mean.dtype, device=self._mean.device)
        if self._has_cov:
            samples = samples.to(self._rot)
            inv_diag = torch.exp(0.5 * self._log_diag)
            samples = samples * inv_diag
            samples = samples @ self._rot.t()
        if isinstance(temperature, torch.Tensor):
            samples = samples * temperature.sqrt()
        else:
            samples = samples * np.sqrt(temperature)
        if self._has_mean:
            samples = samples.to(self._mean)
            samples = samples + self._mean
        return samples

    def _sample(self, n_samples):
        return self._sample_with_temperature(n_samples)


class MeanFreeNormalDistribution(Energy, Sampler):
    """ Mean-free normal distribution. """

    def __init__(self, dim, n_particles, std=1.):
        super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spacial_dims = dim // n_particles
        self.register_buffer("_std", torch.as_tensor(std))

    def _energy(self, x):
        x = self._remove_mean(x).view(-1, self._dim)
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True) / self._std ** 2

    def sample(self, n_samples, temperature=1.) -> torch.Tensor:
        x = torch.ones((n_samples, self._n_particles, self._spacial_dims), dtype=self._std.dtype,
                         device=self._std.device).normal_(mean=0, std=self._std)
        x = self._remove_mean(x)
        x = x.view(-1, self._dim)
        return x

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x
