from oslo.pytorch.activation_checkpointing.checkpoint_function import CheckpointFunction
from oslo.pytorch.activation_checkpointing.checkpoint_partitioner import (
    CheckpointPartitioner,
)
from oslo.pytorch.activation_checkpointing.rng_state_tracker import CudaRNGStatesTracker


class ActivationCheckpointingEngine(object):
    def __init__(
        self,
        num_layers,
        partitioned_checkpointing,
        cpu_checkpointing,
        contiguous_checkpointing,
        mpu=None,
    ):
        rng_tracker = CudaRNGStatesTracker(mpu=mpu)
        partitioner = CheckpointPartitioner(
            mpu=mpu,
            num_layers=num_layers,
            partitioned_checkpointing=partitioned_checkpointing,
            contiguous_checkpointing=contiguous_checkpointing,
            cpu_checkpointing=cpu_checkpointing,
        )
        self.options = {
            "rng_tracker": rng_tracker,
            "partitioner": partitioner,
        }

    def checkpoint(self, function, *args):
        all_outputs = []
        CheckpointFunction.apply(function, self.options, all_outputs, *args)
        if len(all_outputs) == 1:
            return all_outputs
        else:
            return tuple(all_outputs)
