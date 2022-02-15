import json
import os
from typing import Any, Dict, Union

from oslo.pytorch.activation_checkpointing import (
    initialize_activation_checkpointing,
)
from oslo.pytorch.kernel_fusion import initialize_kernel_fusion
from oslo.pytorch.model_parallelism import initialize_model_parallelism
from oslo.pytorch.utils.extenstions import restrict_embedding_resizing


def _type(_type):
    return lambda key, val: {
        "check": isinstance(val, _type),
        "msg": f"``{key}: {val}`` is not a valid set. it must be type of {_type}",
    }


def _one_of_(*args):
    return lambda key, val: {
        "check": val in list(args),
        "msg": f"``{key}: {val}`` is not a valid set. it must be one of the {list(args)}",
    }


SUPPORTED_FEATURES = {
    "model_parallelism": {
        "enable": _type(bool),
        "tensor_parallel_size": _type(int),
        "pipeline_parallel_size": _type(int),
    },
    "activation_checkpointing": {
        "enable": _type(bool),
        "partitioned_checkpointing": _type(bool),
        "contiguous_checkpointing": _type(bool),
        "cpu_checkpointing": _type(bool),
    },
    "kernel_fusion": {
        "enable": _type(bool),
        "memory_efficient_fusion": _type(bool),
        "custom_cuda_kernels": _type(list),
    },
}


def _config_check(arg, user):
    assert len(user) > 0, f"There are no arguments in dictionary."

    if isinstance(user, dict):
        for k in user:
            if isinstance(arg, dict):
                assert k in arg, (
                    f"An argument ``{k}`` is not available. "
                    f"We only support the arguments like {list(arg.keys())}."
                )
            else:
                raise Exception(
                    f"``{k}: {user[k]} is not a valid set. "
                    f"please check your configuration.``"
                )

            if isinstance(user[k], dict):
                _config_check(arg[k], user[k])
            else:
                assert not isinstance(arg[k], dict), (
                    f"``{k}: {user[k]} is not a valid set. "
                    f"please check your configuration.``"
                )

                check_result = arg[k](k, user[k])
                assert check_result["check"], check_result["msg"]
    else:
        raise TypeError("configuration must be type of <class 'dict'>")


def _sanity_check(
    model,
    config: Union[str, Dict[str, Any]],
):
    from transformers import PreTrainedModel

    assert isinstance(
        model, PreTrainedModel
    ), "An argument ``model`` must be the PyTorch Hugging Face Transformers model."

    assert isinstance(config, dict) or isinstance(
        config, str
    ), "An argument ``config`` must be the dictionary object or json path."

    if isinstance(config, str):
        if os.path.isfile(config):
            config = json.load(open(config, encoding="utf-8"))
        else:
            raise ValueError(
                f"Con not find the {config}. "
                f"Please double check your config file name."
            )

    _config_check(SUPPORTED_FEATURES, config)

    return model, config


def initialize(model, config: Union[str, Dict[str, Any]], **kwargs):
    """
    Initialize OSLO engine.

    Args:
        model (PreTrainedModel): The PyTorch Hugging Face Transformers model
        config (Union[str, os.PathLike, Dict[str, Any]]): dict object or json path
    """
    if not hasattr(model, "oslo_initialized"):
        model, config = _sanity_check(model, config)
        model, config = initialize_model_parallelism(model, config, **kwargs)
        model, config = initialize_activation_checkpointing(model, config, **kwargs)
        model, config = initialize_kernel_fusion(model, config, **kwargs)
        model = restrict_embedding_resizing(model)
        setattr(model, "oslo_initialized", True)
    else:
        raise RuntimeError("Can not OSLO initialize twice!")

    return model
