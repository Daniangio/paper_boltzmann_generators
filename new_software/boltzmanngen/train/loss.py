"""
original project = "NequIP" https://github.com/mir-group/nequip
copyright = "2021, MIR"
author = "MIR" (Simon Batzner, Albert Musealian, Lixin Sun, Mario Geiger, Anders Johansson and Tess Smidt)
"""

from typing import Dict, Union, List
import torch
from ._loss import find_loss_function
from torch_runstats import RunningStats, Reduction
from ._keys import ABBREV, LOSS_KEY

class Loss:
    def __init__(
        self,
        params: Union[dict, str, List[str]],
    ):

        self.coeffs = {}
        self.funcs = {}
        self.data_keys = {}
        self.keys = []

        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (str)):
                    func = find_loss_function(value, {})
                    self.coeffs[key] = 1.0
                    self.funcs[key] = func
                elif isinstance(value, (list, tuple)):
                    if len(value) == 2:
                        func = find_loss_function(value[0], {})
                        self.coeffs[key] = value[1]
                        self.funcs[key] = func
                    elif len(value) == 3:
                        assert isinstance(value[2], dict), f"Expected dict in tuple on index 2. Got {type(value[2])}"
                        func = find_loss_function(value[0], value[2])
                        self.coeffs[key] = value[1]
                        self.funcs[key] = func
                    else:
                        raise NotImplementedError(
                            f"Expected tuple of length 2 or 3 (loss_name, weight_coeff, [params]). Got length {len(value)}"
                        )
                else:
                    raise NotImplementedError(
                        f"expected str or tuple. got {type(value)}"
                    )
        else:
            raise NotImplementedError(
                f"loss_params can only be dict. got {type(params)}"
            )

        for key, coeff in self.coeffs.items():
            self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
    
    @classmethod
    def from_config(cls, config):
        loss_params = config.get("loss_params", None)
        instance = cls(params=loss_params)
        energy_models = [func.energy_model for func in instance.funcs.values()]
        return instance, *energy_models

    def __call__(self, pred: dict, temperature: float, direction: str, **kwargs):

        loss = 0.0
        contrib = {}
        for key in self.coeffs:
            key_out, key_jacob, key_direction = key if isinstance(key, tuple) else (key, None, None)
            if key_direction is None or direction == key_direction:
                _loss = self.funcs[key](
                    pred=pred,
                    key=key_out,
                    key_jacob=key_jacob,
                    temperature=temperature,
                    **kwargs
                )
                contrib[key] = _loss
                loss = loss + self.coeffs[key] * _loss
        return loss, contrib


class LossStat:
    """ Accumulates loss values over all batches for each loss component.
    
    """

    def __init__(self):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(),
                reduction=Reduction.MEAN
            )
        }

    def __call__(
        self,
        loss: torch.Tensor,
        loss_contrib: Dict[str, torch.Tensor]
        ):
        """
        Args:

        loss: value of the total loss for the current batch
        loss_contrib: dictionary containing the contribute of each loss component
        """

        results = {}

        results[ABBREV.get(LOSS_KEY)] = self.loss_stat["total"].accumulate_batch(loss).item()

        # go through each component
        for k, v in loss_contrib.items():

            # initialize for the 1st batch
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(
                    dim=tuple(),
                    reduction=Reduction.MEAN
                )
                device = v.get_device()
                self.loss_stat[k].to(device="cpu" if device == -1 else device)

            results[ABBREV.get(LOSS_KEY) + "_" + ABBREV.get(k, k)] = (
                self.loss_stat[k].accumulate_batch(v).item()
            )
        return results

    def reset(self):
        """
        Reset all the counters to zero
        """

        for v in self.loss_stat.values():
            v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)
        return self

    def current_result(self):
        results = {
            ABBREV.get(LOSS_KEY) + "_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results[ABBREV.get(LOSS_KEY)] = self.loss_stat["total"].current_result().item()
        return results