import torch
from typing import Dict, List
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
        log_det_Jzx = torch.log(1e-10 + torch.abs(torch.det(pred[key_jacob])))
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
        hess_values = self.energy_model._hessian(pred[key])
        return hess_values.prod(dim=-1) / (1e-10 + hess_values.abs().prod(dim=-1))
        # probs = self.gausshist(hess_values)
        # return -torch.log(1e-10 + probs)


class PathLoss:
    """ Path Loss
    
    """

    def __init__(
        self,
        energy_model: str,
        params: Dict = {},
        sigmas: List[float] = [1.0, 1.0],
        normalize: bool = True,
        window_size: int = 10,
        logp_eps: List[float] = [1e-2, 1e-2],
        hist_volume_expansions: List[float] = [0.0, 0.0],
    ):
        builder = load_callable(energy_model, module_name="boltzmanngen.distribution")
        self.energy_model = builder(**params)
        self.sigmas = sigmas
        self.normalize = normalize
        self.E_stats = RunningStats(window_size=window_size)
        self.logp_stats = RunningStats(window_size=window_size)
        self.logp_orth_stats = RunningStats(window_size=window_size)

        self.logp_eps = logp_eps
        self.hist_volume_expansions = hist_volume_expansions

    def __call__(
        self,
        pred: dict,
        key: dict,
        bins: int,
        temperature: float = 1.0,
        path_weight: float = 1.0,
        **kwargs
    ):
        E = self.energy_model.energy(pred[key], temperature=temperature)
        if self.normalize:
            self.E_stats.update(E)
        x_start = pred[key][0]
        x_end = pred[key][-1]
        R = x_end - x_start
        centered_x = pred[key] - x_start
        gausshist  = GaussianHistogram(bins=bins, min=0.0-self.hist_volume_expansions[0], max=1.0+self.hist_volume_expansions[0], sigma=self.sigmas[0]).to(x_start.device)
        path_positions = torch.matmul(centered_x, R).div((R).pow(2).sum(0) + 1e-10)
        probs = gausshist(path_positions)
        logp = -torch.log(self.logp_eps[0] + probs)

        R_orth = R.clone()
        R_orth[0], R_orth[1] = -R[1], R[0]
        gausshist_orth  = GaussianHistogram(bins=bins, min=-0.5-self.hist_volume_expansions[1], max=0.5+self.hist_volume_expansions[1], sigma=self.sigmas[1]).to(x_start.device)
        path_positions_orth = torch.matmul(centered_x, R_orth).div((R_orth).pow(2).sum(0) + 1e-10)
        probs_orth = gausshist_orth(path_positions_orth)
        logp_orth = -torch.log(self.logp_eps[1] + probs_orth)
        if self.normalize:
            logp_ = logp.clone()
            logp = logp * self.E_stats.mag_order() / self.logp_stats.mag_order()
            self.logp_stats.update(logp_)
            
            logp_orth_ = logp_orth.clone()
            logp_orth = logp_orth * self.E_stats.mag_order() / self.logp_orth_stats.mag_order()
            self.logp_orth_stats.update(logp_orth_)
        # print(logp.mean(), logp_orth.mean(), E.mean())
        return path_weight * (logp.mean() + logp_orth.mean()) + E.mean()


def find_loss_function(name: str, params: Dict):
    builder = load_callable(name, module_name="boltzmanngen.train")
    return builder(**params)


class GaussianHistogram(torch.nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)
    
    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2)
        x = x.sum(dim=-1)
        x = x.div(1e-10 + x.sum(dim=-1)) # normalization
        return x


class RunningStats:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.items = torch.zeros((self.window_size), dtype=torch.float32)
        self.pointer = 0
        self.device = None
    
    def update(self, item: torch.Tensor):
        if self.device is None:
            self.device = item.device
        self.items[self.pointer] = item.detach().mean()
        self.pointer += 1
        if self.pointer >= self.window_size:
            self.pointer = 0
    
    def mean(self):
        return self.items.mean()
    
    def std(self):
        return self.items.std()
    
    def mag_order(self):
        return torch.tensor(10.0).pow(self.items.mean().abs().log10().int())