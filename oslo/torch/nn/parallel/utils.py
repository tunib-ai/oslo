from typing import Tuple, List

import torch
import torch.nn as nn
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.transformers.modeling_utils import OsloModel


class ParallelWrapper(nn.Module):
    """Marker interface"""


def is_huggingface_model(model: nn.Module):
    try:
        import transformers

        return isinstance(model, transformers.PreTrainedModel)
    except ImportError:
        return False


def is_oslo_model(model: nn.Module):
    if isinstance(model, OsloModel):
        return True

    for module in model.modules():
        if isinstance(module, OsloModel):
            return True
    return False


def is_wrapper(model: nn.Module):
    if isinstance(model, ParallelWrapper):
        return True

    for module in model.modules():
        if isinstance(module, ParallelWrapper):
            return True
    return False


def unwrap_parallel(model: nn.Module):
    while isinstance(model, ParallelWrapper):
        model = model.module
    return model


def get_parameter_dtype(parameter: nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def _update_module_arguments(module: nn.Module, **kwargs):
    for k, v in kwargs.items():
        setattr(module, k, v)


def _remove_module_arguments(module: nn.Module, args: list):
    for k in args:
        delattr(module, k)


def allocate_params(model: nn.Module, parallel_context: ParallelContext):
    for name, parameter in model.named_parameters():
        if hasattr(parameter, "oslo_parallel"):
            device = parallel_context.ranks2device(parameter.oslo_parallel)
            if device is not None:
                parameter.data = parameter.to(
                    f"cuda:{device % parallel_context.local_world_size}"
                )
        else:
            parameter.data = parameter.to(torch.cuda.current_device())

    for name, buffer in model.named_buffers():
        if hasattr(buffer, "oslo_parallel"):
            device = parallel_context.ranks2device(buffer.oslo_parallel)
            if device is not None:
                buffer.data = buffer.to(
                    f"cuda:{device % parallel_context.local_world_size}"
                )
        else:
            buffer.data = buffer.to(torch.cuda.current_device())


def get_parallel_context(module: nn.Module, parallel_context: ParallelContext):
    if parallel_context is None:
        if hasattr(module, "parallel_context"):
            parallel_context = module.parallel_context
        else:
            raise ValueError(
                "Please input parallel context. \n"
                "There are two way to input parallel context: \n"
                "1. model.from_pretrained('model_name', parallel_context=parallel_context) \n"
                "2. model = XXXParallel(model, parallel_context=parallel_context)"
            )

    return parallel_context
