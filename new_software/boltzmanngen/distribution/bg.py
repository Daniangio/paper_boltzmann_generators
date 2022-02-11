from typing import Optional, Union
from boltzmanngen.data import DataConfig
from boltzmanngen.train.loss import Loss
import torch
import numpy as np
from boltzmanngen.distribution import Energy, Sampler
from boltzmanngen.nn._sequential import BaseModule

class BoltzmannGenerator(Energy, Sampler):
    def __init__(
        self,
        prior: Union[Sampler, Energy],
        model: BaseModule,
        target: Energy,
        loss: Loss,
    ):
        """ Constructs Boltzmann Generator, i.e. normalizing flow to sample target density

        Parameters
        ----------
        prior : object
            Prior distribution implementing the energy() and sample() functions
        flow : Flow object
            Flow that can be evaluated forward and reverse
        target : object
            Target distribution implementing the energy() function
        """
        super().__init__(
            target.event_shapes if target is not None else prior.event_shapes
        )
        self._prior = prior
        self._model = model
        self._target = target
        self._loss = loss
        self._device = "cpu"
        self._latest_loss_contrib = None
    
    def to(self, device):
        self._device = device
        self._model = self._model.to(device)
        return self
    
    @property
    def model(self):
        return self._model

    @property
    def prior(self):
        return self._prior
    
    @property
    def loss_contrib(self):
        return self._latest_loss_contrib

    def sample(
        self,
        n_samples,
        temperature=1.0,
    ):
        z = self._prior.sample(n_samples, temperature=temperature)
        data = {
            DataConfig.INPUT_KEY: z.to(self._device)
        }
        data = self._model(data)
        x = data[DataConfig.OUTPUT_KEY]
        dlogp = torch.log(1e-6 + torch.abs(torch.det(data[DataConfig.JACOB_KEY])))

        bg_energy = self._prior.energy(z, temperature=temperature) + dlogp
        target_energy = self._target.energy(x, temperature=temperature)
        log_weights = bg_energy - target_energy
        weights = torch.softmax(log_weights, dim=0).view(-1)

        return {
            "z": z,
            "x": x,
            "dlogp": dlogp,
            "energy": bg_energy,
            "log_weights": log_weights,
            "weights": weights,
        }
    
    def energy(self, x: torch.Tensor, temperature: float = 1.0):
        data = {
            DataConfig.INPUT_KEY: x.to(self._device)
        }
        data = self._model(data, inverse=True)
        loss, loss_contrib = self._loss(pred=data, temperature=temperature, direction=DataConfig.X_TO_Z_KEY)
        self._latest_loss_contrib = loss_contrib
        return loss
    
    def kldiv(self, n_samples: int, temperature: float = 1.0, explore: float = 1.0):
        z = self._prior.sample(n_samples, temperature=temperature)
        data = {
            DataConfig.INPUT_KEY: z.to(self._device)
        }
        data = self._model(data)
        loss, loss_contrib = self._loss(pred=data, temperature=temperature, direction=DataConfig.Z_TO_X_KEY, explore=explore)
        self._latest_loss_contrib = loss_contrib
        return loss
    
    def path(self, n_samples: int, temperature: float = 1.0, path_weight: float = 1.0, x: Optional[torch.Tensor] = None):
        if x is None:
            z = self._prior.sample(2, temperature=temperature)
        else:
            data = {
                DataConfig.INPUT_KEY: x.to(self._device)
            }
            data = self._model(data, inverse=True)
            z = data[DataConfig.OUTPUT_KEY]
        assert z.size()[0] == 2
        z_dim = z.size()[1]
        z_interpolated = torch.zeros((n_samples, z_dim), dtype=torch.torch.get_default_dtype()).to(z.device)
        for i in range(z_dim):
            z_interpolated[:, i] = torch.linspace(z[0, i].item(), z[1, i].item(), n_samples)
        
        data = {
                DataConfig.INPUT_KEY: z_interpolated.to(self._device)
            }
        data = self._model(data)
        loss, loss_contrib = self._loss(pred=data, temperature=temperature, direction=DataConfig.PATH_KEY, bins=int(np.sqrt(n_samples)), path_weight=path_weight)
        self._latest_loss_contrib = loss_contrib
        return loss
    
    def saddle(self, n_samples: int, temperature: float = 1.0):
        z = self._prior.sample(n_samples, temperature=temperature)
        data = {
            DataConfig.INPUT_KEY: z.to(self._device)
        }
        data = self._model(data)
        loss, loss_contrib = self._loss(pred=data, temperature=temperature, direction=DataConfig.SADDLE_KEY)
        self._latest_loss_contrib = loss_contrib
        return loss
    
    def log_weights(self, x, z, dlogp, temperature: float = 1.0, normalize=True):
        bg_energy = self._prior.energy(z, temperature=temperature) + dlogp
        target_energy = self._target.energy(x, temperature=temperature)
        logw = bg_energy - target_energy
        if normalize:
            logw = logw - torch.logsumexp(logw, dim=0)
        return logw.view(-1)
    
    def trigger(self, function_name):
        return self._model.trigger(function_name)