from typing import Any, Dict, Optional
import torch.nn as nn

from boltzmanngen.nn._sequential import BaseModule
from boltzmanngen.data import DataConfig
from boltzmanngen.nn._fullyconnected import FullyConnected


class InvertibleBlock(BaseModule, nn.Module):
    def __init__(
        self,
        parity: int = 0,
        irreps_in: Optional[Dict[str, Any]] = None,
        irreps_out: Optional[Dict[str, Any]] = None,
        resize: BaseModule = FullyConnected,
        translation: BaseModule = FullyConnected,
        **kwargs
    ):
        super().__init__()
        in_field: str = DataConfig.SECOND_CHANNEL_KEY if parity else DataConfig.FIRST_CHANNEL_KEY
        
        transform_in_field: str =  DataConfig.FIRST_CHANNEL_KEY if parity else DataConfig.SECOND_CHANNEL_KEY
        transform_out_field: str = DataConfig.FIRST_CHANNEL_OUT_KEY if parity else DataConfig.SECOND_CHANNEL_OUT_KEY

        self.in_field = in_field
        self.out_field = in_field
        if irreps_out is None:
            irreps_out = {self.out_field: irreps_in[self.in_field]}

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[in_field],
            irreps_out=irreps_out,
        )

        self._resize = resize(
            in_field = transform_in_field,
            out_field = transform_out_field,
            irreps_in = self.irreps_in,
            irreps_out = {transform_out_field: self.irreps_out[self.out_field]}
        )

        self._translation = translation(
            in_field = transform_in_field,
            out_field = transform_out_field,
            irreps_in = self.irreps_in,
            irreps_out = {transform_out_field: self.irreps_out[self.out_field]}
        )

        assert self._resize.out_field == self._translation.out_field
        self.transform_out_field = self._translation.out_field


    def forward(self, data: DataConfig.Type, inverse: bool = False) -> DataConfig.Type:
        funcs = [self.resize_data, self.translate_data]
        for func in reversed(funcs) if inverse else funcs:
            data = func(data, inverse)
        return data

    def translate_data(self, data, inverse):
        data = self._translation(data)
        translation = -data[self.transform_out_field] if inverse else data[self.transform_out_field]
        data[self.out_field] = data[self.in_field] + translation
        return data

    def resize_data(self, data, inverse):
        data = self._resize(data)
        resize = 1/data[self.transform_out_field] if inverse else data[self.transform_out_field]
        data[self.out_field] = data[self.in_field] * resize
        return data
