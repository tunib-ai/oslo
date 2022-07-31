import json
import logging
from copy import deepcopy
import torch
from typing import List, Tuple
from dataclasses import dataclass
from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import (
    PipelineParallel,
    TensorParallel,
    ShardedDataParallel,
    FullyShardedDataParallel,
    DistributedDataParallel,
)

NoneType = type(None)


def _type(_dtype):
    return lambda key, val: {
        "check": isinstance(val, _dtype),
        "msg": f"{key}: {val} is not a valid set. it must be type of {_dtype}",
    }


def _values(*args):
    return lambda key, val: {
        "check":
            val in args,
        "msg":
            f"{key}: {val} is not a valid set. it must be one of {list(args)}",
    }


def _dp_config(*args):
    params = {}
    for arg in args:
        params.update(DATA_PARALLEL_CONFIGS[arg])
    return params


DATA_PARALLEL_CONFIGS = {
    "ddp": {
        "broadcast_buffers": _type(bool),
        "bucket_cap_mb": _type(int),
        "find_unused_parameters": _type(bool),
        "check_reduction": _type(bool),
        "gradient_as_bucket_view": _type(bool),
    },
    "oss": {
        "broadcast_fp16": _type(bool),
        "force_broadcast_object": _type(bool),
    },
    "sdp": {
        "broadcast_buffers": _type(bool),
        "sync_models_at_startup": _type(bool),
        "reduce_buffer_size": _type(int),
        "auto_refresh_trainable": _type(bool),
        "reduce_fp16": _type(bool),
        "warn_on_trainable_params_changed": _type(bool),
    },
    "fsdp": {
        "disable_reshard_on_root": _type((bool, NoneType)),
        "fp32_reduce_scatter": _type((bool, NoneType)),
        "flatten_parameters": _type((bool, NoneType)),
        "move_params_to_cpu": _type((bool, NoneType)),
        "move_grads_to_cpu": _type((bool, NoneType)),
        "bucket_cap_mb": _type((int, NoneType)),
        "no_broadcast_optim_state": _type((bool, NoneType)),
        "clear_autocast_cache": _type(bool),
        "force_input_to_fp32": _type(bool),
        "state_dict_on_rank_0_only": _type(bool),
        "offload_config": {
            "offload_type": _type((str, NoneType)),
            "dir": _type((str, NoneType)),
        },
    },
}

DATA_PARALLEL_CONFIGS_BY_ZERO_STAGE = {
    0: _dp_config("ddp"),
    1: _dp_config("sdp", "oss"),
    2: _dp_config("fsdp", "oss"),
    3: _dp_config("fsdp", "oss"),
}

SUPPORTED_FEATURES = {
    "mixed_precision": {
        "enable": _type(bool),
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": _type(bool),
        "contiguous_checkpointing": _type(bool),
        "cpu_checkpointing": _type(bool),
    },
    "sequence_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "params": {
            "broadcast_buffers": _type(bool),
            "bucket_cap_mb": _type(int),
            "find_unused_parameters": _type(bool),
            "check_reduction": _type(bool),
            "gradient_as_bucket_view": _type(bool),
        },
    },
    "data_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "zero_stage": _values(0, 1, 2, 3),
        "params": lambda stage: DATA_PARALLEL_CONFIGS_BY_ZERO_STAGE[stage],
    },
    "tensor_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "parallel_mode": _values("1d", "2d", "2.5d", "3d"),
        "params": {
            "parallel_depth_2.5d": _type(int),
        },
    },
    "pipeline_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "params": {
            "memory_computation_balance": _type(float),
        },
    },
    "expert_parallelism": {
        "enable": _type(bool),
        "parallel_size": _type(int),
        "params": {
            "top_k": _type(int),
            "capacity_factor_train": _type(int),
            "capacity_factor_eval": _type(int),
            "select_policy": _values("first", "random"),
            "noisy_policy": _values("jitter", "gaussian"),
            "drop_tokens": _type(bool),
            "use_rts": _type(bool),
            "use_residual": _type(bool),
        },
    },
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


@dataclass
class Config:

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.mkconfig(self)

    @staticmethod
    def mkconfig(obj):
        for k, v in obj.__dict__.items():
            if isinstance(v, dict):
                obj.__setattr__(k, Config(**v))

    def __getitem__(self, item):
        if item not in self.__dict__:
            return None
        else:
            getattr(self, item)

    def __repr__(self):
        return str(self.__dict__.items())


class OsloTrainerConfig(Config):
    """
    This object contains a Oslo feature configuration dictionary

    [Oslo `TrainingArguments`] uses this class to set oslo features includes parallel, fused optimizer etc.
    json file or dictionary form should be like the following:
        {
            "mixed_precision": {
                "enable": _type(bool),
            },
            "activation_checkpointing": {
                "partitioned_checkpointing": _type(bool),
                "contiguous_checkpointing": _type(bool),
                "cpu_checkpointing": _type(bool),
            },
            "sequence_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "params": {
                    "broadcast_buffers": _type(bool),
                    "bucket_cap_mb": _type(int),
                    "find_unused_parameters": _type(bool),
                    "check_reduction": _type(bool),
                    "gradient_as_bucket_view": _type(bool),
                },
            },
            "data_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "zero_stage": _values(0, 1, 2, 3),
                "params": lambda stage: DATA_PARALLEL_CONFIGS_BY_ZERO_STAGE[stage],
            },
            "tensor_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "parallel_mode": _values("1d", "2d", "2.5d", "3d"),
                "params": {
                    "parallel_depth_2.5d": _type(int),
                },
            },
            "pipeline_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "params": {
                    "memory_computation_balance": _type(float),
                },
            },
            "expert_parallelism": {
                "enable": _type(bool),
                "parallel_size": _type(int),
                "params": {
                    "top_k": _type(int),
                    "capacity_factor_train": _type(int),
                    "capacity_factor_eval": _type(int),
                    "select_policy": _values("first", "random"),
                    "noisy_policy": _values("jitter", "gaussian"),
                    "drop_tokens": _type(bool),
                    "use_rts": _type(bool),
                    "use_residual": _type(bool),
                },
            },
        }

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to Oslo user config file or dict.

    """

    def __init__(self, config_file_or_dict):
        super(OsloTrainerConfig, self).__init__()
        self.cpu_offload = False

        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overridden
            cfg = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with open(config_file_or_dict, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            raise ValueError(
                "expecting either a path to a oslo config file or a dict")
        _config_check(SUPPORTED_FEATURES, cfg)
        super(OsloTrainerConfig, self).__init__(**cfg)

        logging.info("*** OSLO CONFIG ***")
        if not all([self['mixed_precision'], self.mixed_precision["enable"]]):
            self.mixed_precision = None
        else:
            logging.info("mixed_precision: enabled")

        if not all([self['data_parallelism'], self.data_parallelism["enable"]]):
            self.data_parallelism = None
        else:
            if self.data_parallelism['parallel_size'] is None:
                logging.warning(
                    "data_parallelism can not be usable because parallel_size is required."
                )
                self.data_parallelism = None

            elif self.data_parallelism['zero_stage'] is None:
                logging.warning(
                    "data_parallelism can not be usable because zero_stage is required."
                )
                self.data_parallelism = None
            else:
                logging.info(
                    f"data_parallelism: enabled\n\tparallel_size: {self.data_parallelism['parallel_size']}\n\tzero_stage: {self.data_parallelism['zero_stage']}"
                )
                if hasattr(self.data_parallelism, 'params') and self.data_parallelism.params['cpu_offload']:
                    self.cpu_offload = True

        if not all([self['sequence_parallelism'], self.sequence_parallelism["enable"]]):
            self.sequence_parallelism = None
        else:
            if self.sequence_parallelism['parallel_size'] is None:
                logging.warning(
                    "sequence_parallelism can not be usable because parallel_size is required."
                )
                self.sequence_parallelism = None
            else:
                logging.info(
                    f"sequence_parallelism: enabled\n\tparallel_size: {self.sequence_parallelism['parallel_size']}"
                )

        if not all(
            [self['tensor_parallelism'], self.tensor_parallelism["enable"]]):
            self.tensor_parallelism = None
        else:
            if self.tensor_parallelism['parallel_size'] is None:
                logging.warning(
                    "tensor_parallelism can not be usable because parallel_size is required."
                )
                self.tensor_parallelism = None
            elif self.tensor_parallelism['parallel_mode'] is None:
                logging.warning(
                    "tensor_parallelism can not be usable because parallel_mode is required."
                )
                self.tensor_parallelism = None
            else:
                logging.info(
                    f"tensor_parallelism: enabled\n\tparallel_size: {self.tensor_parallelism['parallel_size']}\n\tparallel_mode: {self.tensor_parallelism['parallel_mode']}"
                )

        if not all(
            [self['pipeline_parallelism'], self.pipeline_parallelism["enable"]
            ]):
            self.pipeline_parallelism = None
        else:
            if self.pipeline_parallelism['parallel_size'] is None:
                logging.warning(
                    "pipeline_parallelism can not be usable because parallel_size is required."
                )
                self.pipeline_parallelism = None
            else:
                logging.info(
                    f"pipeline_parallelism: enabled\n\tparallel_size: {self.pipeline_parallelism['parallel_size']}"
                )

        if not all(
            [self['expert_parallelism'], self.expert_parallelism["enable"]]):
            self.expert_parallelism = None
        else:
            if self.expert_parallelism['parallel_size'] is None:
                logging.warning(
                    "expert_parallelism can not be usable because parallel_size is required."
                )
                self.expert_parallelism = None
            else:
                logging.info(
                    f"expert_parallelism: enabled\n\tparallel_size: {self.expert_parallelism['parallel_size']}"
                )


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
    data_parallel_size = cfg.data_parallelism.parallel_size if cfg.data_parallelism else 1
    sequence_parallel_size = cfg.sequence_parallelism.parallel_size if cfg.sequence_parallelism else 1
    expert_parallel_size = cfg.expert_parallelism.parallel_size if cfg.expert_parallelism else 1
    pipeline_parallel_size = cfg.pipeline_parallelism.parallel_size if cfg.pipeline_parallelism else 1
    tensor_parallel_size = cfg.tensor_parallelism.parallel_size if cfg.tensor_parallelism else 1

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=data_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_depth=cfg.tensor_parallelism.param['parallel_depth_2.5d'] if cfg.tensor_parallelism['param'] else 1,
        tensor_parallel_mode=cfg.tensor_parallelism.parallel_mode if cfg.tensor_parallelism else "1d"
    )

    if tensor_parallel_size > 1 and sequence_parallel_size > 1:
        raise AttributeError(
            "TensorParallel and SequenceParallel can't be used at the same time. Modify oslo config to avoid wrong parallel setting"
        )

    model_wrapper = []
    if data_parallel_size > 1:
        if cfg.data_parallelism.zero_stage == 0:
            model_wrapper.append(DistributedDataParallel)
        elif cfg.data_parallelism.zero_stage == 1:
            model_wrapper.append(ShardedDataParallel)
        elif cfg.data_parallelism.zero_stage == 2:
            model_wrapper.append(FullyShardedDataParallel)
        elif cfg.data_parallelism.zero_stage == 3:
            model_wrapper.append(FullyShardedDataParallel)
        else:
            logging.warning(
                "No delivered stage for data_parallelism, default mode DistributedDataParallel is set"
            )
            model_wrapper.append(DistributedDataParallel)
    if tensor_parallel_size > 1:
        model_wrapper.append(TensorParallel)
    if pipeline_parallel_size > 1:
        model_wrapper.append(PipelineParallel)
    # TODO expert mode

    return parallel_context, model_wrapper
