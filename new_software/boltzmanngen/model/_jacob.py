from boltzmanngen.nn._sequential import BaseModule
from boltzmanngen.nn._grad_output import GradientOutput

from boltzmanngen.data import DataConfig


def ModelJacobian(model: BaseModule) -> GradientOutput:
    r""" Compute the Jacobian of the model

    Args:
        model: the model to wrap.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """
    
    if DataConfig.JACOB_KEY in model.irreps_out:
        raise ValueError("This model already has Jacobian output.")
    return GradientOutput(
        func=model,
        of=DataConfig.OUTPUT_KEY,
        wrt=DataConfig.INPUT_KEY,
        out_field=DataConfig.JACOB_KEY,
        sign=1.0,
    )
