from oslo.transformers.oslo_init import OsloTrainerConfig


oslo_init_dict_form = {
    "data_parallelism": {
        "zero_stage": "2",
        "distributed_data_parallel": False,
        "data_parallel_size": 1,
        "sequence_parallel_size": 1
    },
    "model_parallelism": {
        "expert_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "tensor_parallel_depth": 1,
        "tensor_parallel_mode": "1d"
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": False,
        "contiguous_checkpointing": False
    },
    "kernel_fusion": {
        "memory_efficient_fusion": False
    },
    "lazy_initialization": False,
    "backend": "nccl"
}

user_config_from_dict = OsloTrainerConfig(oslo_init_dict_form)

user_config_from_json = OsloTrainerConfig("oslo_user_config.json")

