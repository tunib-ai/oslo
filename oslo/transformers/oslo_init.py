import json
import logging
from copy import deepcopy
import torch
from typing import List, Tuple

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
    ShardedDataParallel,
    FullyShardedDataParallel,
    DistributedDataParallel,
    SequenceDataParallel,
)


def _type(_type):
    return lambda key, val: {
        "check":
            isinstance(val, _type),
        "msg":
            f"``{key}: {val}`` is not a valid set. it must be type of {_type}",
    }


SUPPORTED_FEATURES = {
    "data_parallelism": {
        "data_parallel_size": _type(int),
        "sequence_parallel_size": _type(int),
        "zero2": {
            "broadcast_buffers": _type(bool),
            "sync_models_at_startup": _type(bool),
            "reduce_buffer_size": _type(int),
            "auto_refresh_trainable": _type(bool),
            "reduce_fp16": _type(bool),
            "warn_on_trainable_params_changed": _type(bool),
        },
        "zero3": {
            "reshard_after_forward": _type((bool, type(None))),
            "disable_reshard_on_root": _type((bool, type(None))),
            "mixed_precision": _type((bool, type(None))),
            "fp32_reduce_scatter": _type((bool, type(None))),
            "flatten_parameters": _type((bool, type(None))),
            "move_params_to_cpu": _type((bool, type(None))),
            "compute_dtype": _type((torch.dtype, type(None))),
            "buffer_dtype": _type((torch.dtype, type(None))),
            "move_grads_to_cpu": _type((bool, type(None))),
            "bucket_cap_mb": _type((int, type(None))),
            "compute_device": _type((torch.device, type(None))),
            "no_broadcast_optim_state": _type((bool, type(None))),
            "state_dict_device": _type((torch.device, type(None))),
            "clear_autocast_cache": _type(bool),
            "force_input_to_fp32": _type(bool),
            "verbose": _type(bool),
            "cpu_offload": _type((bool, type(None))),
            "offload_config": {
                "offload_type": _type((str, type(None))),
                "dir": _type((str, type(None))),
            },
            "state_dict_on_rank_0_only": _type(bool),
        },
        "distributed": {
            "device_ids": _type((list, torch.device, type(None))),
            "output_device": _type((int, torch.device, type(None))),
            "dim": _type(int),
            "broadcast_buffers": _type(bool),
            "bucket_cap_mb": _type(int),
            "find_unused_parameters": _type(bool),
            "check_reduction": _type(bool),
            "gradient_as_bucket_view": _type(bool)
        },
        "sequence": {
            "device_ids": _type((list, torch.device, type(None))),
            "output_device": _type((int, torch.device, type(None))),
            "dim": _type(int),
            "broadcast_buffers": _type(bool),
            "bucket_cap_mb": _type(int),
            "find_unused_parameters": _type(bool),
            "check_reduction": _type(bool),
            "gradient_as_bucket_view": _type(bool)
        },
    },
    "model_parallelism": {
        "expert_parallel_size": _type(int),
        "pipeline_parallel_size": _type(int),
        "tensor_parallel_size": _type(int),
        "tensor_parallel_depth": _type(int),
        "tensor_parallel_mode": _type(str),
        "pipline": {
            "memory_computation_balance": _type(float),
            "tracing_inputs": _type((dict, type(None))),
        },
        "tensor": {
            "mapping": _type((dict, type(None)))
        },
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": _type(bool),
        "contiguous_checkpointing": _type(bool),
        "cpu_checkpointing": _type(bool),
    },
    "lazy_initialization": _type(bool),
    "backend": _type(str),
}

TENSOR_PARALLEL_MODE_TYPES = {
    "1d": ParallelMode.TENSOR_1D,
    "2d": ParallelMode.TENSOR_2D,
    "2.5d": ParallelMode.TENSOR_2P5D,
    "3d": ParallelMode.TENSOR_3D,
}


def _config_check(arg, user_config):
    # assert len(user_config) > 0, "There are no arguments in dictionary."

    if isinstance(user_config, dict):
        for k in user_config:
            if isinstance(arg, dict):
                assert k in arg, (
                    f"An argument ``{k}`` is not available. "
                    f"We only support the arguments like {list(arg.keys())}.")
            else:
                raise Exception(f"``{k}: {user_config[k]} is not a valid set. "
                                f"please check your configuration.``")

            if isinstance(user_config[k], dict):
                _config_check(arg[k], user_config[k])
            else:
                assert not isinstance(arg[k], dict), (
                    f"``{k}: {user_config[k]} is not a valid set. "
                    f"please check your configuration.``")
                check_result = arg[k](k, user_config[k])
                assert check_result["check"], check_result["msg"]
    else:
        raise TypeError("configuration must be type of <class 'dict'>")


def check_user_config(user_config):
    _config_check(SUPPORTED_FEATURES, user_config)
    if "model_parallelism" in user_config and "tensor_parallel_mode" in user_config[
            "model_parallelism"]:
        assert (
            user_config["model_parallelism"]["tensor_parallel_mode"]
            in TENSOR_PARALLEL_MODE_TYPES.keys()
        ), (f"{user_config['model_parallelism']['tensor_parallel_mode']} is not valid type of tensor_parallel_mode. "
            f"choose one of {', '.join(TENSOR_PARALLEL_MODE_TYPES.keys())}")
        # check user config tensor_parallel_mode
        user_tp_mode = user_config["model_parallelism"]["tensor_parallel_mode"]
        user_config["model_parallelism"][
            "tensor_parallel_mode"] = TENSOR_PARALLEL_MODE_TYPES[user_tp_mode]


class OsloTrainerConfig:
    """
    This object contains a Oslo feature configuration dictionary

    [Oslo `TrainingArguments`] uses this class to set oslo features includes parallel, fused optimizer etc.
    json file or dictionary form should be like the following:
        {
            "data_parallelism": {
                "data_parallel_size": _type(int),
                "sequence_parallel_size": _type(int),
                "zero2": {
                    "broadcast_buffers": _type(bool),
                    "sync_models_at_startup": _type(bool),
                    "reduce_buffer_size": _type(int),
                    "auto_refresh_trainable": _type(bool),
                    "reduce_fp16": _type(bool),
                    "warn_on_trainable_params_changed": _type(bool),
                },
                "zero3": {
                    "reshard_after_forward": _type((bool, type(None))),
                    "disable_reshard_on_root": _type((bool, type(None))),
                    "mixed_precision": _type((bool, type(None))),
                    "fp32_reduce_scatter": _type((bool, type(None))),
                    "flatten_parameters": _type((bool, type(None))),
                    "move_params_to_cpu": _type((bool, type(None))),
                    "compute_dtype": _type((torch.dtype, type(None))),
                    "buffer_dtype": _type((torch.dtype, type(None))),
                    "move_grads_to_cpu": _type((bool, type(None))),
                    "bucket_cap_mb": _type((int, type(None))),
                    "compute_device": _type((torch.device, type(None))),
                    "no_broadcast_optim_state": _type((bool, type(None))),
                    "state_dict_device": _type((torch.device, type(None))),
                    "clear_autocast_cache": _type(bool),
                    "force_input_to_fp32": _type(bool),
                    "verbose": _type(bool),
                    "cpu_offload": _type((bool, type(None))),
                    "offload_config": {
                        "offload_type": _type((str, type(None))),
                        "dir": _type((str, type(None))),
                    },
                    "state_dict_on_rank_0_only": _type(bool),
                },
                "distributed": {
                    "device_ids": _type((list, torch.device, type(None))),
                    "output_device": _type((int, torch.device, type(None))),
                    "dim": _type(int),
                    "broadcast_buffers": _type(bool),
                    "bucket_cap_mb": _type(int),
                    "find_unused_parameters": _type(bool),
                    "check_reduction": _type(bool),
                    "gradient_as_bucket_view": type(bool),
                },
                "sequence": {
                    "device_ids": _type((list, torch.device, type(None))),
                    "output_device": _type((int, torch.device, type(None))),
                    "dim": _type(int),
                    "broadcast_buffers": _type(bool),
                    "bucket_cap_mb": _type(int),
                    "find_unused_parameters": _type(bool),
                    "check_reduction": _type(bool),
                    "gradient_as_bucket_view": type(bool),
                },
            },
            "model_parallelism": {
                "expert_parallel_size": _type(int),
                "pipeline_parallel_size": _type(int),
                "tensor_parallel_size": _type(int),
                "tensor_parallel_depth": _type(int),
                "tensor_parallel_mode": _type(str),
                "pipline": {
                    "memory_computation_balance": _type(float),
                    "tracing_inputs": _type(dict, type(None)),
                },
                "tensor": {"mapping": _type(dict, type(None))},
            },
            "activation_checkpointing": {
                "partitioned_checkpointing": _type(bool),
                "contiguous_checkpointing": _type(bool),
                "cpu_checkpointing": _type(bool),
            },
            "lazy_initialization": _type(bool),
            "backend": _type(str),
        }

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to Oslo user config file or dict.

    """

    DP = "data_parallelism"
    MP = "model_parallelism"
    ACTIVE_CHECKPOINT = "activation_checkpointing"
    LAZY_INIT = "lazy_initialization"
    BACKEND = "backend"

    def __init__(self, config_file_or_dict):
        self._dtype = None
        self.train_batch_size = None

        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            config = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError(
                "expecting either a path to a oslo config file or a dict")

        check_user_config(config)
        self.config = config
        self._set_config(config)

    def __repr__(self):
        return str(self.config.items())

    def _set_config(self, config: dict):
        for k, v in config.items():
            # if isinstance(v, dict):
            #     self._set_config(v)
            # else:
            #     setattr(self, k, v)
            setattr(self, k, v)

    def get_value(self, key):
        """
        Returns a value matching to the key
        """
        if key not in self.config.keys():
            raise KeyError(f"Check {key} is valid name of OsloTrainerConfig.")
        return self.config.get(key)

    def dtype(self):
        if self._dtype is None:
            raise ValueError(
                "trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype


def init_oslo_features(
    oslo_init_config: OsloTrainerConfig,) -> Tuple[ParallelContext, List]:
    """
    Init OSLO features with json or dict configuration user passed.
    ParallelContext or other effective features should be defined on this function
    and Trainer could use this outputs

    This function returns two object, ParallelContext and WrapperModule from user config
    TrainArgumet class use this to re-define model
    >> model = ...
    >> parallel_context = ParallelContext.from_torch(...)
    >> wrapper_model = TensorParallel(model, parallel_context)
    >> allocate_params(wrapper_model, parallel_context)
    """

    cfg = oslo_init_config
    dp_cfg, mp_cfg = {}, {}
    if hasattr(cfg, "data_parallelism"):
        dp_cfg = cfg.data_parallelism
    if hasattr(cfg, "model_parallelism"):
        mp_cfg = cfg.model_parallelism

    data_parallel_size = dp_cfg["data_parallel_size"] if "data_parallel_size" in dp_cfg else 1
    sequence_parallel_size = dp_cfg["sequence_parallel_size"] if "sequence_parallel_size" in dp_cfg else 1
    expert_parallel_size = mp_cfg["expert_parallel_size"] if "expert_parallel_size" in mp_cfg else 1
    pipeline_parallel_size = mp_cfg["pipeline_parallel_size"] if "pipeline_parallel_size" in mp_cfg else 1
    tensor_parallel_size = mp_cfg["tensor_parallel_size"] if "tensor_parallel_size" in mp_cfg else 1

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=data_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_depth=mp_cfg["tensor_parallel_depth"] if "tensor_parallel_depth" in mp_cfg else 1,
        tensor_parallel_mode=mp_cfg["tensor_parallel_mode"] if "tensor_parallel_mode" in mp_cfg else TENSOR_PARALLEL_MODE_TYPES["1d"],
        backend=cfg.backend if hasattr(cfg, "backend") else "nccl",
        seed=cfg.seed if hasattr(cfg, "seed") else 42,
    )

    if tensor_parallel_size > 1 and sequence_parallel_size > 1:
        raise AttributeError(
            "TensorParallel and SequenceParallel can't not be used at the same time. Modify oslo config to avoid wrong parallel setting"
        )

    model_wrapper = []
    if data_parallel_size > 1:
        if "distributed" in dp_cfg:
            model_wrapper.append(DistributedDataParallel)
        elif "zero2" in dp_cfg:
            model_wrapper.append(FullyShardedDataParallel)
        elif "zero3" in dp_cfg:
            model_wrapper.append(ShardedDataParallel)
        else:
            logging.warning(
                "No delivered stage for data_parallelism, default mode DistributedDataParallel is set"
            )
            model_wrapper.append(DistributedDataParallel)
        if "zero2" in dp_cfg and "zero3" in dp_cfg:
            logging.warning(
                "Both ZeRO2 and ZeRO3 are set, Ignore ZeRO2 and apply ZeRO3.")

    if tensor_parallel_size > 1:
        model_wrapper.append(TensorParallel)
    if pipeline_parallel_size > 1:
        model_wrapper.append(PipelineParallel)
    # TODO expert mode

    return parallel_context, model_wrapper
