import inspect
from boltzmanngen.utils.instantiate import load_callable
from boltzmanngen.nn._sequential import BaseModule


def model_from_config(
    config,
) -> BaseModule:
    """Build a model based on `config`.

    Args:
        config: Configuration file

    Returns:
        The built model.
    """

    # Build
    builders = [
        load_callable(b, module_name="boltzmanngen.model")
        for b in config.get("model_builders", [])
    ]

    model = None

    for builder_i, builder in enumerate(builders):
        pnames = inspect.signature(builder).parameters
        params = {}
        if "config" in pnames:
            params["config"] = config
        if "model" in pnames:
            if builder_i == 0:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the model as an input, but it's the first builder so there is no model to provide"
                )
            params["model"] = model
        else:
            if builder_i > 0:
                raise RuntimeError(
                    f"All model_builders but the first one must take the model as an argument; {builder.__name__} doesn't"
                )
        model = builder(**params)
        if not isinstance(model, BaseModule):
            raise TypeError(
                f"Builder {builder.__name__} didn't return a BaseModule, got {type(model)} instead"
            )

    return model
