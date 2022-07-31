from oslo.transformers.oslo_init import OsloTrainerConfig, init_oslo_features


oslo_init_dict_form = {
    "data_parallelism": {
        "enable": True,
        "parallel_size": 4,
        "zero_stage": 0,
    }
}

user_config_from_dict = OsloTrainerConfig(oslo_init_dict_form)

# user_config_from_json = OsloTrainerConfig(
#     "tests/transformers/trainer/oslo_user_config.json"
# )

print(user_config_from_dict)

res = init_oslo_features(user_config_from_dict)

print(res)
