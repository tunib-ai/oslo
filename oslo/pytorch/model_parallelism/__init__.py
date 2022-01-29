from functools import partial

from oslo.pytorch.model_parallelism.network.mpu import MPU
from oslo.pytorch.model_parallelism.utils.mappings import TPMapping
from oslo.pytorch.model_parallelism.tensor_parallel_enigne import (
    TensorParallelEngine,
    TensorDeparallelEngine,
)
from oslo.pytorch.model_parallelism.utils.extensions import (
    from_parallelized,
    save_parallelized,
    resize_token_embeddings,
)


def initialize_model_parallelism(model, config, **kwargs):
    if "model_parallelism" in config:
        mp_config = config["model_parallelism"]
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
                mapping = kwargs.pop("tp_mapping", TPMapping())
                tensor_parallel_engine = TensorParallelEngine(model, mpu, mapping)
                tensor_parallel_engine.parallelize()
                setattr(
                    model,
                    "resize_token_embeddings",
                    partial(resize_token_embeddings, self=model),
                )

            setattr(model, "from_parallelized", partial(from_parallelized, self=model))
            setattr(model, "save_parallelized", partial(save_parallelized, self=model))

    return model, config
