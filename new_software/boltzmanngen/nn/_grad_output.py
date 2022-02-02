"""
original project = "NequIP" https://github.com/mir-group/nequip
copyright = "2021, MIR"
author = "MIR" (Simon Batzner, Albert Musealian, Lixin Sun, Mario Geiger, Anders Johansson and Tess Smidt)
"""

from typing import List, Union, Optional
from boltzmanngen.data import DataConfig

import torch

from e3nn.util.jit import compile_mode

from boltzmanngen.nn._sequential import BaseModule


@compile_mode("script")
class GradientOutput(BaseModule, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient.

    Args:
        func: the model to wrap
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
        out_field: the field in which to return the computed gradients. Defaults to ``f"d({of})/d({wrt})"`` for each field in ``wrt``.
        sign: either 1 or -1; the returned gradient is multiplied by this.
    """
    sign: float

    def __init__(
        self,
        func: BaseModule,
        of: str,
        wrt: str,
        out_field: str = None,
        sign: float = 1.0,
    ):
        super().__init__()
        assert sign in (1.0, -1.0)
        self.sign = sign
        self.of = of
        self.wrt = wrt
        self.func = func
        self.out_field = out_field or f"d({of})/d({wrt})"

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_out,
            irreps_out=func.irreps_in,
        )

        self.irreps_out.update(
            {of: self.irreps_in[wrt]}
        )

    def forward(self, data: DataConfig.Type, inverse: bool = False) -> DataConfig.Type:
        # set req grad
        old_requires_grad = data[self.wrt].requires_grad
        data[self.wrt].requires_grad_(True)
        
        # run func
        data = self.func(data, inverse=inverse)

        b, N = data[self.of].size()
        grads = []
        for i in range(N):
            output = torch.zeros(b, N)
            output[:, i] = 1.
            output = output.requires_grad_(True).to(data[self.of].get_device())
            grads.append(torch.autograd.grad(data[self.of], data[self.wrt], grad_outputs=output, create_graph=True, retain_graph=True)[0])

        jacT = torch.stack(grads, dim=-1)
        data[self.out_field] = self.sign * torch.transpose(jacT, 1, 2)

        # unset requires_grad_
        data[self.wrt].requires_grad_(old_requires_grad)

        return data
