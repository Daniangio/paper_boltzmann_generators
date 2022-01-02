from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from torch.nn import Sequential, Module

from boltzmanngen.data import DataConfig
from boltzmanngen.utils.instantiate import instantiate


class BaseModule:
    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        my_irreps_in: Dict[str, Any] = {},
        irreps_out: Dict[str, Any] = {},
    ):
        """ Setup the expected data fields and their irreps for this module.

        Args:
            irreps_in (dict): maps names of all input fields from previous modules or
                data to their corresponding irreps
            required_irreps_in: sequence of names of fields that must be present in
                ``irreps_in``, but that can have any irreps.
            my_irreps_in (dict): maps names of fields to the irreps they must have for
                this graph module. Will be checked for consistancy with ``irreps_in``
            irreps_out (dict): mapping names of fields that are modified/output by
                this graph module to their irreps.
        """

        irreps_in = {} if irreps_in is None else irreps_in

        irreps_in = DataConfig._fix_irreps_dict(irreps_in)
        my_irreps_in = DataConfig._fix_irreps_dict(my_irreps_in)
        irreps_out = DataConfig._fix_irreps_dict(irreps_out)

        # Confirm compatibility:
        # with required_irreps_in
        for k in required_irreps_in:
            if k not in irreps_in:
                raise ValueError(
                    f"This {type(self)} requires field '{k}' to be in irreps_in"
                )
        # with my_irreps_in
        for k in my_irreps_in:
            if k in irreps_in and irreps_in[k] != my_irreps_in[k]:
                raise ValueError(
                    f"The given input irreps {irreps_in[k]} for field '{k}' is incompatible with this configuration {type(self)}; should have been {my_irreps_in[k]}"
                )
        
        # Save stuff
        self.irreps_in = irreps_in
        
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out


class SequentialNetwork(BaseModule, Sequential):
    def __init__(
        self,
        modules: Union[Sequence[Module], Dict[str, Module]],
    ):
        if isinstance(modules, dict):
            module_list = list(modules.values())
        else:
            module_list = list(modules)
        
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:]):
            assert DataConfig._irreps_compatible(m1.irreps_out, m2.irreps_in)
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        # torch.nn.Sequential will name children correctly if passed an OrderedDict
        if isinstance(modules, dict):
            modules = OrderedDict(modules)
        else:
            modules = OrderedDict((f"module_{i}", m) for i, m in enumerate(module_list))
        super().__init__(modules)
    
    def forward(self, input: DataConfig.Type, inverse: bool = False) -> DataConfig.Type:
        for module in reversed(self) if inverse else self:
            input = module(input, inverse)
        return input
    
    @classmethod
    def from_parameters(
        cls,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
        shared_params: Mapping,
        irreps_in: Optional[dict] = None,
        **kwargs
    ):
        """ Construct a ``SequentialModule`` of modules built from a shared set of parameters.
        """
        
        built_modules = []
        for name, builder in layers.items():
            if not isinstance(name, str):
                raise ValueError(f"`'name'` must be a str; got `{type(name)}`")
            if isinstance(builder, tuple):
                builder, params = builder
                if not isinstance(params, Dict):
                    raise ValueError(f"If ``layers`` value is a Tuple, the second element represents the \
                                       ``params`` of the first element (Callable). It must be a Dict; got `{type(params)}`")
            else:
                params = {}
            if not callable(builder):
                raise TypeError(
                    f"The builder has to be a class or a function. got {type(builder)}"
                )

            instance = instantiate(
                builder=builder,
                prefix=name,
                positional_args=(
                    dict(
                        irreps_in=(
                            built_modules[-1].irreps_out
                            if len(built_modules) > 0
                            else irreps_in
                        )
                    )
                ),
                optional_args=params,
                shared_args=shared_params,
            )

            if not isinstance(instance, BaseModule):
                raise TypeError(
                    f"Builder `{builder}` for layer with name `{name}` did not return a BaseModule, instead got a {type(instance).__name__}"
                )

            built_modules.append(instance)

        return cls(
            OrderedDict(zip(layers.keys(), built_modules)),
        )