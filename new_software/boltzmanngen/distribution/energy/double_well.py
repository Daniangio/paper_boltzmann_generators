"""
original project = "bgflow" https://github.com/noegroup/bgflow
copyright = MIT License
author = Jonas Köhler, Andreas Krämer, Manuel Dibak, Leon Klein, Frank Noé
"""

import torch

from .base import Energy


class DoubleWellEnergy(Energy):
    def __init__(self, dim, a=0, b=-2.0, c=0.5):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x, just_dimer: bool = False):
        d = x[:, [0]]
        v = x[:, 1:]
        dimer_energy = (self._a * d + self._b * d.pow(2) + self._c * d.pow(4))
        oscillator = 0.0 if just_dimer else 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return (dimer_energy + oscillator).view(-1)


class ShiftedDoubleWellEnergy(Energy):
    def __init__(self, dim, a=0, b=-2.0, c=0.5):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x, **kwargs):
        X1 = x[:, [0]]
        X2 = x[:, 1:]
        energy = (self._a * X1 + self._b * X1.pow(2) + self._c * X1.pow(4)) * 10 * torch.sin(X2 + 1) \
                 + 10 * torch.sin(X2) + 0.01 * X1.pow(10) + 0.01 * X2.pow(10)
        return (energy).view(-1)
    
    def _hessian(self, x: torch.Tensor):
        X1 = x[:, [0]]
        X2 = x[:, 1:]
        hessian = torch.zeros((len(X1), self.dim, self.dim), dtype=torch.float32).to(x.device)
        hessian[:, 0, 0] = ((2 * self._b + 12 * self._c * X1.pow(2)) * 10 * torch.sin(X2 + 1) + 90 * X1.pow(8)).view(-1)
        hessian[:, 0, 1] = ((2 * self._b * X1 + 4 * self._c * X1.pow(3)) * 10 * torch.cos(X2 + 1)).view(-1)
        hessian[:, 1, 0] = ((self._a + 2 * self._b * X1 + 4 * self._c * X1.pow(3)) * 10 * torch.cos(X2 + 1)).view(-1)
        hessian[:, 1, 1] = (-(self._a * X1 + self._b * X1.pow(2) + self._c * X1.pow(4)) * 10 * torch.sin(X2 + 1) - 10 * torch.sin(X2) + 90 * X2.pow(8)).view(-1)
        return torch.linalg.eigvals(hessian).real
        # e_vals = []
        # for hess in hessian:
        #     e_vals.append(torch.linalg.eigvals(hess).real)
        # return torch.stack(e_vals)


class MultiDimensionalDoubleWell(Energy):
    def __init__(self, dim, a=0.0, b=-4.0, c=1.0, transformer=None):
        super().__init__(dim)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        self.register_buffer("_c", c)
        if transformer is not None:
            self.register_buffer("_transformer", transformer)
        else:
            self._transformer = None

    def _energy(self, x):
        if self._transformer is not None:
            x = torch.matmul(x, self._transformer)
        e1 = self._a * x + self._b * x.pow(2) + self._c * x.pow(4)
        return e1.sum(dim=1, keepdim=True)


class MultiDoubleWellPotential(Energy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via
    .. math:: E_{DW}(d) = a * (d-offset)^4 + b * (d-offset)^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of Lennard-Jones particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset):
        super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset
        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)

def compute_distances(x, n_particles, n_dimensions, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, n_dimensions)
    distances = torch.cdist(x, x)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances