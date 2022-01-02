import logging
from boltzmanngen.data import DataConfig
from e3nn.o3 import Irreps

from boltzmanngen.nn._sequential import SequentialNetwork
from boltzmanngen.nn._fullyconnected import FullyConnected


def InvertibleModel(config) -> SequentialNetwork:
    logging.debug("Start building the network model")

    num_layers = config.get("num_layers", 3)
    irreps_in={DataConfig.SECOND_CHANNEL_KEY: Irreps([(16, (0, 1))])}

    layers = {

    }

    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer_{layer_i}"] = (
            FullyConnected,
            dict(
                in_field=DataConfig.SECOND_CHANNEL_KEY,
                out_field=DataConfig.SECOND_CHANNEL_OUT_KEY,
            )
        )

    return SequentialNetwork.from_parameters(
        shared_params=config,
        layers=layers,
        irreps_in=irreps_in
    )