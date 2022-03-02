from oslo.pytorch.model_parallelism.tensor_parallel_enigne import (
    TensorParallelEngine,
    TensorDeparallelEngine,
)
from oslo.pytorch.model_parallelism.utils.distributed import allocate, deallocate


class ModelParallelEngine(object):
    """3D model parallel engine"""

    def __init__(self, model, mpu, tp_mapping):
        self.model = model
        self.mpu = mpu
        self.tp_mapping = tp_mapping

    def allocate(self):
        for parameter in self.model.parameters():
            allocate(self.mpu, parameter)
        for parameter in self.model.buffers():
            allocate(self.mpu, parameter)

    def parallelize(self):
        TensorParallelEngine(self.model, self.mpu, self.tp_mapping).parallelize()
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
