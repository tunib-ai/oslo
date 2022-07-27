from oslo.transformers.oslo_init import OsloTrainerConfig, init_oslo_features


oslo_init_dict_form = {
    "data_parallelism": {
        "data_parallel_size": 2,
        "sequence_parallel_size": 2,
        "zero2": {
            "broadcast_buffers": True,
            "sync_models_at_startup": True,
            "reduce_buffer_size": 0,
            "auto_refresh_trainable": True,
            "reduce_fp16": True,
            "warn_on_trainable_params_changed": True,
        },
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": False,
        "contiguous_checkpointing": False,
    },
    "lazy_initialization": False,
    "backend": "nccl",
}

user_config_from_dict = OsloTrainerConfig(oslo_init_dict_form)

user_config_from_json = OsloTrainerConfig(
    "tests/transformers/trainer/oslo_user_config.json"
)

print(user_config_from_dict)

res = init_oslo_features(user_config_from_dict)

print(res)
