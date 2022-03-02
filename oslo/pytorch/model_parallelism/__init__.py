from functools import partial


def initialize_model_parallelism(model, config, **kwargs):
    if "model_parallelism" in config:
        mp_config = config["model_parallelism"]
        force_gpu = config["commons"]["force_gpu"]

        if force_gpu is False:
            raise ValueError(
                "``force_gpu=False`` is not compatible with model parallelism"
            )

        if "enable" in mp_config and mp_config["enable"] is True:
            from oslo.pytorch.model_parallelism.network.mpu import MPU
            from oslo.pytorch.model_parallelism.model_parallel_engine import (
                ModelParallelEngine,
                ModelDeparallelEngine,
            )
            from oslo.pytorch.model_parallelism.utils.extensions import (
                from_parallelized,
                save_parallelized,
            )

            tp_size = mp_config.get("tensor_parallel_size", 1)
            pp_size = mp_config.get("pipeline_parallel_size", 1)
            deployment_mode = mp_config.get("deployment_mode", False)

            assert tp_size >= 1, "param `tensor_parallel_size` must be positive."
            assert pp_size >= 1, "param `pipeline_parallel_size` must be positive."
            assert (
                tp_size & (tp_size - 1) == 0
            ), "param `tensor_parallel_size` must be power of 2."

            if tp_size * pp_size > 1:
                if tp_size > 1:
                    assert model.config.num_attention_heads % tp_size == 0, (
                        "``tensor_parallel_size`` must be divisible by ``num_attention_heads``. "
                        "Please check your model configuration."
                    )
                    assert model.config.hidden_size % tp_size == 0, (
                        "``tensor_parallel_size`` must be divisible by ``hidden_size``. "
                        "Please check your model configuration."
                    )
                    assert model.config.num_attention_heads >= tp_size, (
                        "``tensor_parallel_size`` must be same or greather than ``num_attention_heads``. "
                        "Please check your model configuration."
                    )
                    assert model.config.hidden_size >= tp_size, (
                        "``tensor_parallel_size`` must be same or greather than ``hidden_size``. "
                        "Please check your model configuration."
                    )

                master_addr = config.get("master_addr", "localhost")
                master_port = config.get("master_port", 29500)
                tp_mapping = kwargs.pop("tp_mapping", None)

                if deployment_mode is False:
                    mpu = MPU(
                        tensor_parallel_size=tp_size,
                        pipeline_parallel_size=pp_size,
                        master_addr=master_addr,
                        master_port=master_port,
                    )
                    mp_engine = ModelParallelEngine(model, mpu, tp_mapping)
                    mp_engine.parallelize()

                else:
                    from oslo.pytorch.model_parallelism.deployment_engine import (
                        DeploymentEngine,
                    )

                    deployment_engine = DeploymentEngine(
                        model,
                        tp_mapping=tp_mapping,
                        tp_size=tp_size,
                        pp_size=pp_size,
                        master_addr=master_addr,
                        master_port=master_port,
                        seed=config.get("seed", None),
                    )
                    deployment_engine.parallelize()

                setattr(
                    model, "from_parallelized", partial(from_parallelized, self=model)
                )
                setattr(
                    model, "save_parallelized", partial(save_parallelized, self=model)
                )

    return model, config
