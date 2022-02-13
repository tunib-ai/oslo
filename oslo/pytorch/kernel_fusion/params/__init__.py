import inspect
import itertools
from abc import abstractmethod

import torch

from oslo.pytorch.kernel_fusion.utils import is_iterable

_CACHE_FOR_MAPPING = {}


class TranslatedParams(object):
    def __init__(
        self,
        module_name=None,
        module_cls=None,
        model_cls=None,
        params=None,
        method=None,
    ):
        self.module_name = module_name
        self.module_cls = module_cls
        self.model_cls = model_cls
        self.params = params
        self.method = method

    def __eq__(self, other):
        return (
            other.module_name == self.module_name
            and other.module_cls == self.module_cls
            and other.model_cls == self.model_cls
        )

    def key(self):
        return self.module_name, self.module_cls, self.model_cls


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def register_params(fn, module_name=None, module_cls=None, model_cls=None):
    class_name = ".".join(fn.__qualname__.split(".")[:-1])
    if class_name not in _CACHE_FOR_MAPPING:
        _CACHE_FOR_MAPPING[class_name] = [
            TranslatedParams(
                module_name=module_name,
                module_cls=module_cls,
                model_cls=model_cls,
                params=fn,
            )
        ]
    else:
        _CACHE_FOR_MAPPING[class_name].append(
            TranslatedParams(
                module_name=module_name,
                module_cls=module_cls,
                model_cls=model_cls,
                params=fn,
            )
        )


class TensorMeta(object):
    def __init__(self, *vals, dtype=None, necessary=False):
        self.size = torch.Size(vals)
        self._size = vals
        self.dtype = dtype
        self.necessary = necessary

    def __str__(self):
        return f"Size({self._size})"

    def __repr__(self):
        return f"Size({self._size})"


class ModuleManager(object):
    def get_modules(self, model, module_name=None, module_cls=None, model_cls=None):
        modules = []
        assert module_name is not None or module_cls is not None, (
            f"You must input module name or class name. "
            f"current module name: {module_name}, class name: {module_cls}."
        )

        if model_cls is not None:
            if not model_cls == model.__class__.__qualname__:
                return modules

        for name, module in model.named_modules():
            if module_cls is None and module_name is not None:
                if name in module_name:
                    modules.append(module)
            elif module_name is None and module_cls is not None:
                if module.__class__.__qualname__ == module_cls:
                    modules.append(module)
            else:
                if module_name in name and module.__class__.__qualname__ == module_cls:
                    modules.append(module)

        return modules

    def get_module(self, model, module_name=None, module_cls=None, model_cls=None):
        modules = self.get_modules(
            model=model,
            module_name=module_name,
            module_cls=module_cls,
            model_cls=model_cls,
        )
        return modules[0] if len(modules) > 0 else None

    @staticmethod
    def get_param_dict(*args, **kwargs):
        param_dict = {}
        orig_forward_parameters = kwargs.pop("orig_forward_parameters", {})

        for param_name, param_value in orig_forward_parameters.items():
            for input_name, input_value in kwargs.items():
                if input_name == param_name:
                    if input_value == param_value.default:
                        param_dict[param_name] = (input_value, True)
                    else:
                        param_dict[param_name] = (input_value, False)

        arg_list = list(args)

        if len(arg_list) != 0:
            for param_name, param_value in orig_forward_parameters.items():
                if len(arg_list) == 0:
                    break
                if param_name not in param_dict:
                    input_value = arg_list[0]
                    if input_value == param_value.default:
                        param_dict[param_name] = (input_value, True)
                    else:
                        param_dict[param_name] = (input_value, False)
                    arg_list = arg_list[1:]

        return param_dict


class Params(ModuleManager):
    """
    Hold the information of input parameters of model
    """

    def __init__(self, model, bsz=1, seq_len=1):
        self.model = model
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = model.config
        self.n_layers = self.config.num_hidden_layers

        self.bsz = bsz
        self.seq_len = seq_len
        self.tp_size = (
            self.model.mpu.get_tensor_parallel_world_size()
            if hasattr(self.model, "mpu")
            else 1
        )

        self._translated_params_material = [
            TranslatedParams(
                module_cls=m.module_cls,
                module_name=m.module_name,
                model_cls=m.model_cls,
                params=m.params(self),
                method=self,
            )
            for m in _CACHE_FOR_MAPPING[self.__class__.__qualname__]
        ]

    @staticmethod
    @abstractmethod
    def supported_args():
        raise NotImplementedError

    @property
    def hid_size(self):
        return self.config.hidden_size // self.tp_size

    @property
    def head_size(self):
        return self.hid_size // self.num_heads

    @property
    def num_heads(self):
        predefined_names = ["num_attention_heads", "n_head", "num_heads"]

        for predefined_name in predefined_names:
            if hasattr(self.config, predefined_name):
                return getattr(self.config, predefined_name) // self.tp_size

        raise ValueError("Can not find the number of head from config object !")

    @property
    def max_len(self):
        predefined_names = ["max_position_embeddings", "n_positions"]

        for predefined_name in predefined_names:
            if hasattr(self.config, predefined_name):
                return getattr(self.config, predefined_name)

        raise ValueError("Can not find max length from config object !")

    @property
    def type_vocab_size(self):
        predefined_names = ["type_vocab_size"]

        for predefined_name in predefined_names:
            if hasattr(self.config, predefined_name):
                return getattr(self.config, predefined_name)

        raise ValueError("Can not find type vocab size from config object !")

    def translated_params(self):
        translated_params = {}
        for params in self._translated_params_material:
            module = self.get_module(
                self.model,
                module_name=params.module_name,
                module_cls=params.module_cls,
                model_cls=params.model_cls,
            )

            if module is not None:
                allowed_params = inspect.signature(module.forward).parameters
                allowed_params_list = list(allowed_params.keys())

                for name in params.params:
                    if (name not in allowed_params_list) and (
                        "kwargs" not in allowed_params_list
                    ):
                        raise ValueError(
                            f"param name ``{name}`` can not be used for class {params.module_cls}. "
                            f"allowed parameters: {allowed_params_list}."
                        )

                all_available_param = []
                for name, param in params.params.items():
                    if isinstance(param, dict):
                        raise ValueError(
                            "Can not use dictionary of ``TensorMeta()``. "
                            "Please input available available list of tensor meta type or "
                            "nested tensor meta type (List[TensorMeta], Tuple[TensorMeta], ..."
                        )
                    elif is_iterable(param):
                        for _param in param:
                            all_available_param.append((name, _param))
                    else:
                        raise ValueError(
                            f"value of available parameters must be iterable (list or tuple). "
                            f"but you input is {param}."
                        )

                non_optional_params = [
                    name
                    for name, value in allowed_params.items()
                    if value.default == inspect._empty
                ]

                combinations = []
                for i in range(0, len(all_available_param) + 1):
                    for subset in itertools.combinations(all_available_param, i):
                        combinations.append(subset)

                all_necessary_params = [
                    k for k, v in params.params.items() if v[0].necessary is True
                ]

                input_candidates = []
                for combination in combinations:
                    has_non_default_params = len(non_optional_params) == 0
                    for elem in combination:
                        if elem[0] in non_optional_params:
                            has_non_default_params = True
                            break

                    if len(combination) != 0 and has_non_default_params:
                        elem_names = [elem[0] for elem in combination]
                        if len(elem_names) == len(set(elem_names)):
                            if set(all_necessary_params) <= set(elem_names):
                                input_candidates.append({k: v for k, v in combination})

                translated_params[module] = TranslatedParams(
                    module_cls=params.module_cls,
                    module_name=params.module_name,
                    model_cls=params.model_cls,
                    params=input_candidates,
                    method=self,
                )

        return translated_params
