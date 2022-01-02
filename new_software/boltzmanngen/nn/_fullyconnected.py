from typing import Any, Dict, Optional
import torch.nn as nn

from boltzmanngen.nn._sequential import BaseModule
from boltzmanngen.data import DataConfig


class FullyConnected(BaseModule, nn.Module):
    def __init__(
        self,
        in_field: str = DataConfig.SECOND_CHANNEL_KEY,
        out_field: Optional[str] = DataConfig.SECOND_CHANNEL_OUT_KEY,
        irreps_in: Optional[Dict[str, Any]] = None,
        irreps_out: Optional[Dict[str, Any]] = None,
        hidden_layers: int = 2,
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        self.in_field = in_field
        self.out_field = out_field if out_field is not None else in_field
        if irreps_out is None:
            irreps_out = {
                out_field: irreps_in[in_field]
            }

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[in_field],
            irreps_out=irreps_out,
        )

        modules = [
            nn.Linear(self.irreps_in[in_field].dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(hidden_layers - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_dim, self.irreps_out[out_field].dim))

        self._fc = nn.Sequential(*modules)

    def forward(self, data: DataConfig.Type) -> DataConfig.Type:
        data[self.out_field] = self._fc(data[self.in_field])
        return data
