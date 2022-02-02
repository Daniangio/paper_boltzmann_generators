from typing import Any, Dict, List
from e3nn.util.jit import compile_mode
import torch
from e3nn.o3 import Irreps

from boltzmanngen.nn._sequential import BaseModule
from boltzmanngen.data import DataConfig


@compile_mode("script")
class IOEncoding(BaseModule, torch.nn.Module):
    num_types: int
    set_features: bool

    def __init__(
        self,
        irreps_in: Dict[str, Any],
        field_in: List[str] = DataConfig.INPUT_KEY,
        fields_io: List[str] = [
            DataConfig.FIRST_CHANNEL_KEY,
            DataConfig.SECOND_CHANNEL_KEY
        ],
        field_out: List[str] = DataConfig.OUTPUT_KEY,
        initial: bool = True,
        **kwargs
    ):
        super().__init__()
        self.field_in = field_in
        self.fields_io = fields_io
        self.field_out = field_out
        self.initial = initial

        irreps_out = {
            self.field_out: irreps_in[self.field_in] # * 2
        }
        for field_io in self.fields_io:
            irreps_out[field_io] = Irreps([(irreps_in[self.field_in].dim // 2, (0, 1))]) # irreps_in[self.field_in]
        
        irreps_in[self.field_in] = irreps_in[self.field_in] # * 2

        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: DataConfig.Type, inverse: bool = False):
        if not (self.initial ^ inverse): # XNOR
            data[self.field_out] = torch.cat((data[self.fields_io[0]], data[self.fields_io[1]]), dim=1)
        else:
            half_length = data[self.field_in].size()[1] // 2
            data[self.fields_io[0]] = data[self.field_in][:, :half_length]
            data[self.fields_io[1]] = data[self.field_in][:, half_length:]
        return data