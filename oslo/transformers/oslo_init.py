import importlib.util
import json
import logging
from copy import deepcopy
import torch
from typing import List

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.nn.parallel.data_parallel.sequence_data_parallel import (
    SequenceDataParallel,)


from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
    ShardedDataParallel,
    FullyShardedDataParallel,
    DistributedDataParallel,
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
        "stage": _type(str),
        "data_parallel_size": _type(int),
        "sequence_parallel_size": _type(int),
    },
    "model_parallelism": {
        "expert_parallel_size": _type(int),
        "pipeline_parallel_size": _type(int),
        "tensor_parallel_size": _type(int),
        "tensor_parallel_depth": _type(int),
        "tensor_parallel_mode": _type(str),
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": _type(bool),
        "contiguous_checkpointing": _type(bool),
        "cpu_checkpointing": _type(bool),
    },
    "kernel_fusion": {
        "memory_efficient_fusion": _type(bool),
        "custom_cuda_kernels": _type(list),
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
    assert len(user_config) > 0, f"There are no arguments in dictionary."

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
    assert user_config['model_parallelism'][
        'tensor_parallel_mode'] in TENSOR_PARALLEL_MODE_TYPES.keys(), (
            f"{user_config['model_parallelism']['tensor_parallel_mode']} is not valid type of tensor_parallel_mode. "
            f"choose one of {', '.join(TENSOR_PARALLEL_MODE_TYPES.keys())}")


class OsloTrainerConfig:
    """
    This object contains a Oslo feature configuration dictionary

    [Oslo `TrainingArguments`] uses this class to set oslo features includes parallel, fused optimizer etc.
    json file or dictionary form should be like the following:
        {
            "data_parallelism": {
                "stage": _type(str),
                "data_parallel_size": _type(int),
                "sequence_parallel_size": _type(int),
            },
            "model_parallelism": {
                "expert_parallel_size": _type(int),
                "pipeline_parallel_size": _type(int),
                "tensor_parallel_size": _type(int),
                "tensor_parallel_depth": _type(int),
                "tensor_parallel_mode": _type(str),
            },
            "activation_checkpointing": {
                "partitioned_checkpointing": _type(bool),
                "contiguous_checkpointing": _type(bool),
                "cpu_checkpointing": _type(bool),
            },
            "kernel_fusion": {
                "memory_efficient_fusion": _type(bool),
                "custom_cuda_kernels": _type(list),
            },
            "lazy_initialization": _type(bool),
            "backend": _type(str),
            "seed": _type(int)
        }

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    DP = "data_parallelism"
    MP = "model_parallelism"
    ACTIVE_CHECKPOINT = "activation_checkpointing"
    KERNEL_FUSION = "kernel_fusion"
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

        # check user config tensor_parallel_mode
        user_tp_mode = config['model_parallelism']['tensor_parallel_mode']
        config['model_parallelism'][
            'tensor_parallel_mode'] = TENSOR_PARALLEL_MODE_TYPES[user_tp_mode]
        # check user config data_parallel dp_stage

        self._set_config(config)
        self.config = config

    def __repr__(self):
        return self.config.__repr__()

    def _set_config(self, config: dict):
        for k, v in config.items():
            if isinstance(v, dict):
                self._set_config(v)
            else:
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

    def adjust_train_args(self, args):
        self.train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        # if args.train_batch_size != train_batch_size:


def init_oslo_features(oslo_init_config: OsloTrainerConfig) -> (ParallelContext, List):
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

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=cfg.data_parallel_size,
        sequence_parallel_size=cfg.sequence_parallel_size,
        expert_parallel_size=cfg.expert_parallel_size,
        pipeline_parallel_size=cfg.pipeline_parallel_size,
        tensor_parallel_size=cfg.tensor_parallel_size,
        tensor_parallel_depth=cfg.tensor_parallel_depth,
        tensor_parallel_mode=cfg.tensor_parallel_mode,
        backend=cfg.backend,
        seed=cfg.seed,
    )

    if cfg.tensor_parallel_size > 1 and cfg.sequence_parallel_size > 1:
        raise AttributeError(
            "TensorParallel and SequenceParallel can't not be used at the same time. Modify oslo config to avoid wrong parallel setting"
        )
    model_wrapper = []
    if cfg.tensor_parallel_size > 1:
        model_wrapper.append(TensorParallel)
    if cfg.sequence_parallel_size > 1:
        model_wrapper.append(SequenceDataParallel)
    if cfg.pipeline_parallel_size > 1:
        model_wrapper.append(PipelineParallel)
    if cfg.data_parallel_size > 1:
        if hasattr(cfg, 'stage'):
            if cfg.stage == 'zero2':
                model_wrapper.append(ShardedDataParallel)
            elif cfg.stage == 'zero3':
                model_wrapper.append(FullyShardedDataParallel)
            elif cfg.stage == 'ddp':
                model_wrapper.append(DistributedDataParallel)
            else:
                raise AttributeError(
                    "OSLO supports zero stage 1&2 and 3 that are respectively ShardedDataParallel and FullyShardedDataParallel."
                )
        else:
            logging.warning("No delivered stage for data_parallelism, default mode DistributedDataParallel is set")
            model_wrapper.append(DistributedDataParallel)

    return parallel_context, model_wrapper
