from oslo.pytorch.model_parallelism.pipeline_parallel_engine import (
    PipelineParallelEngine,
)
from oslo.pytorch.model_parallelism.tensor_parallel_engine import (
    TensorDeparallelEngine,
    TensorParallelEngine,
)
from oslo.pytorch.model_parallelism.utils.distributed import (
    allocate,
    deallocate,
)


class ModelParallelEngine(object):
    """3D model parallel engine"""

    def __init__(
        self,
        model,
        mpu,
        tp_mapping,
        tracing_inputs,
        memory_computation_balance_factor,
    ):
        self.model = model
        self.mpu = mpu
        self.tp_mapping = tp_mapping
        self.tracing_inputs = tracing_inputs
        self.memory_computation_balance_factor = memory_computation_balance_factor

    def allocate(self):
        for parameter in self.model.parameters():
            allocate(self.mpu, parameter)
        for parameter in self.model.buffers():
            allocate(self.mpu, parameter)

    def parallelize(self):
        TensorParallelEngine(
            self.model,
            self.mpu,
            self.tp_mapping,
        ).parallelize()

        PipelineParallelEngine(
            self.model,
            self.mpu,
            self.tracing_inputs,
            self.memory_computation_balance_factor,
        ).parallelize()

        self.allocate()


class ModelDeparallelEngine(object):
    """3D model deparallel engine"""

    def __init__(self, model, mpu, tp_mapping):
        self.model = model
        self.mpu = mpu
        self.tp_mapping = tp_mapping

    def deallocate(self):
        for parameter in self.model.parameters():
            deallocate(parameter)
        for parameter in self.model.buffers():
            deallocate(parameter)

    def deparallelize(self):
        TensorDeparallelEngine(self.model, self.mpu, self.tp_mapping).deparallelize()
        self.deallocate()
