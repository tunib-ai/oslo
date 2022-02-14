from functools import partial


def initialize_model_parallelism(model, config, **kwargs):
    if "model_parallelism" in config:
        mp_config = config["model_parallelism"]

        if "enable" in mp_config and mp_config["enable"] is True:
            from oslo.pytorch.model_parallelism.network.mpu import MPU
            from oslo.pytorch.model_parallelism.tensor_parallel_enigne import (
                TensorDeparallelEngine,
                TensorParallelEngine,
            )
            from oslo.pytorch.model_parallelism.utils.extensions import (
                from_parallelized,
                save_parallelized,
            )
            from oslo.pytorch.model_parallelism.utils.mappings import (
                TensorParallelismMapping,
            )

            tp_size = mp_config.get("tensor_parallel_size", 1)
            pp_size = mp_config.get("pipeline_parallel_size", 1)

            assert tp_size >= 1, "param `tensor_parallel_size` must be positive."
            assert pp_size >= 1, "param `pipeline_parallel_size` must be positive."
            assert (
                tp_size & (tp_size - 1) == 0
            ), "param `tensor_parallel_size` must be power of 2."

            if tp_size * pp_size > 1:
                mpu = MPU(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)

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

                    mapping = kwargs.pop("tp_mapping", TensorParallelismMapping())
                    tensor_parallel_engine = TensorParallelEngine(model, mpu, mapping)
                    tensor_parallel_engine.parallelize()

                setattr(
                    model, "from_parallelized", partial(from_parallelized, self=model)
                )
                setattr(
                    model, "save_parallelized", partial(save_parallelized, self=model)
                )

    return model, config
