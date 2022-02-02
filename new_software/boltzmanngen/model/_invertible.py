import logging
from boltzmanngen.data import DataConfig
from e3nn.o3 import Irreps

from boltzmanngen.nn._sequential import SequentialNetwork
from boltzmanngen.nn._invblock import InvertibleBlock
from boltzmanngen.nn._encode_io import IOEncoding


def InvertibleModel(config) -> SequentialNetwork:
    logging.debug("Start building the network model")

    num_layers = config.get("num_layers", 3)
    input_dim = config.get("input_dim", 2)
    irreps_in={
        DataConfig.INPUT_KEY: Irreps([(input_dim, (0, 1))]),
    }

    layers = {
        "encode_input": IOEncoding
    }

    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer_{layer_i}"] = (
            InvertibleBlock,
            {
                "parity": layer_i%2
            }
        )
    
    layers.update(
        {
            "encode_output": (
            IOEncoding,
            {
                "initial": False
            }
        )
        }
    )

    return SequentialNetwork.from_parameters(
        shared_params=config,
        layers=layers,
        irreps_in=irreps_in
    )