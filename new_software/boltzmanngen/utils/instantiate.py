from typing import Callable, List, Optional, Union
from importlib import import_module


def instantiate(
    builder: Callable,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: dict = {},
    shared_args: dict = {},
):
    try:
        instance = builder(**positional_args, **optional_args, **shared_args)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build object with prefix `{prefix}` using builder `{builder.__name__}`"
        ) from e
    
    return instance

def load_callable(obj: Union[str, Callable], module_name: Optional[str] = None) -> Callable:
    """Load a callable from a name, or pass through a callable."""
    if callable(obj):
        pass
    elif isinstance(obj, str):
        obj = getattr(import_module(module_name), obj)
    else:
        raise TypeError
    assert callable(obj), f"{obj} isn't callable"
    return obj