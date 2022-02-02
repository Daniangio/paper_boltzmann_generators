import torch
from typing import Dict
from boltzmanngen.utils.instantiate import load_callable


class JKLLoss:
    """ KL divergenge loss. "Train by energy" component of the loss function.
    
    """

    def __init__(
        self,
        energy_model: str,
        params: Dict = {}
    ):
        builder = load_callable(energy_model, module_name="boltzmanngen.distribution")
        self.energy_model = builder(**params)

    def __call__(
        self,
        pred: dict,
        key: dict,
        key_jacob: str,
        temperature: float = 1.0,
        explore: float = 1.0
    ):
        E = self.energy_model.energy(pred[key], temperature=temperature)
        log_det_Jzx = torch.log(1e-6 + torch.abs(torch.det(pred[key_jacob])))
        return -explore * log_det_Jzx + E


class MLLoss:
    """ Maximum Likelyhood loss. "Train by example" component of the loss function.
    
    """

    def __init__(
        self,
        energy_model: str,
        params: Dict = {}
    ):
        builder = load_callable(energy_model, module_name="boltzmanngen.distribution")
        self.energy_model = builder(**params)

    def __call__(
        self,
        pred: Dict,
        key: Dict,
        key_jacob: str,
        temperature: float = 1.0,
    ):
        E = self.energy_model.energy(pred[key], temperature=temperature)
        log_det_Jxz = torch.log(1e-6 + torch.abs(torch.det(pred[key_jacob])))
        return -log_det_Jxz + E


class HessianLoss:
    """ Hessian Loss
    
    """

    def __init__(
        self,
        energy_model: str,
        params: Dict = {}
    ):
        builder = load_callable(energy_model, module_name="boltzmanngen.distribution")
        self.energy_model = builder(**params)

    def __call__(
        self,
        pred: dict,
        key: dict,
        **kwargs
    ):
        return self.energy_model._hessian(pred[key]).prod(dim=-1)


def find_loss_function(name: str, params: Dict):
    builder = load_callable(name, module_name="boltzmanngen.train")
    return builder(**params)