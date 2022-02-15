import inspect
from copy import deepcopy
from logging import getLogger

import torch
import torch.distributed as dist

from oslo.pytorch.kernel_fusion.mem_efficient.compilers import (
    ts_compile,
    default_decompositions,
    memory_efficient_fusion,
)
from oslo.pytorch.kernel_fusion.mem_efficient.model_output import OutputManager
from oslo.pytorch.kernel_fusion.mem_efficient.partitioners import (
    min_cut_rematerialization_partition,
)

logger = getLogger(__name__)


class TensorMeta(object):
    def __init__(self, *sizes, dtype):
        self.sizes = sizes
        self.dtype = dtype

    def __str__(self):
        return f"Size{self.sizes}"

    def __repr__(self):
        return f"Size{self.sizes}"

    def __eq__(self, other):
        return other.sizes == self.sizes and other.dtype == self.dtype


def is_iterable(elem):
    try:
        iter(elem)
        return True
    except:
        return False


class MemoryEfficientFusionEngine(object):
    __compiling_info_cache__ = []

    def __init__(self, model):
        self.model = model

    def fuse(self):
        OutputManager(self.model).register_model_output_classes()
        self.register_forward()
        return memory_efficient_fusion(
            self.model,
            min_cut_rematerialization=True,
        )

    @staticmethod
    def logging_rank_0(message):
        if dist.is_initialized() and dist.is_available():
            if dist.get_rank() == 0:
                logger.warning(message)
        else:
            logger.warning(message)

    @staticmethod
    def get_param_dict(*args, **kwargs):
        param_dict = {}
        orig_forward_parameters = kwargs.pop("orig_forward_parameters", {})
        for param_name, param_value in orig_forward_parameters.items():
            for input_name, input_value in kwargs.items():
                if isinstance(input_value, torch.Tensor):
                    if input_name == param_name:
                        param_dict[input_name] = input_value

        arg_list = list(args)
        if len(arg_list) != 0:
            for param_name, param_value in orig_forward_parameters.items():
                if len(arg_list) == 0:
                    break
                if param_name not in param_dict:
                    input_value = arg_list[0]
                    if isinstance(input_value, torch.Tensor):
                        param_dict[param_name] = input_value
                    arg_list = arg_list[1:]

        return param_dict

    def tensor2meta(self, values):
        if isinstance(values, torch.Tensor):
            values = TensorMeta(*values.size(), dtype=values.dtype)
        elif is_iterable(values):
            if isinstance(values, dict):
                values = {k: self.tensor2meta(v) for k, v in values.items()}
            else:
                values = [self.tensor2meta(v) for v in values]

        return values

    def register_forward(self):
        forward_fn = self.model.forward
        forward_parameters = deepcopy(dict(inspect.signature(forward_fn).parameters))

        def new_forward_fn(*args, **kwargs):
            param_dict = self.get_param_dict(
                *args,
                **kwargs,
                orig_forward_parameters=forward_parameters,
            )

            meta = self.tensor2meta(param_dict)

            if meta not in self.__compiling_info_cache__:
                self.__compiling_info_cache__.append(meta)
                self.logging_rank_0(
                    f"[MemoryEfficientFusion] Compiling new graph for {meta}."
                )

            return forward_fn(**param_dict)

        self.model.forward = new_forward_fn
