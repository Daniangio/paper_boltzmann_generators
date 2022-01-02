import logging
from boltzmanngen.data import DataConfig
from e3nn.o3 import Irreps

from boltzmanngen.nn._sequential import SequentialNetwork
from boltzmanngen.nn._invblock import InvertibleBlock


def InvertibleModel(config) -> SequentialNetwork:
    logging.debug("Start building the network model")

    num_layers = config.get("num_layers", 3)
    irreps_in={
        DataConfig.FIRST_CHANNEL_KEY: Irreps([(4, (0, 1))]),
        DataConfig.SECOND_CHANNEL_KEY: Irreps([(4, (0, 1))])
    }

    layers = {

    }

    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer_{layer_i}"] = (
            InvertibleBlock,
            {
                "parity": layer_i%2
            }
        )

    return SequentialNetwork.from_parameters(
        shared_params=config,
        layers=layers,
        irreps_in=irreps_in
    )