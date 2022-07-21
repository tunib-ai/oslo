from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.utils import get_parallel_context


class PipelineParallel(nn.Module):
    """
    Pipeline parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
        memory_computation_balance (float): memory computation balance factor

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`.
        2. Support multiple scheduling algorithms.
        3. Support inter-module partitioning described in Sagemaker Model Parallelism.

    Examples:
        >>> from oslo.torch.nn.parallel import PipelineParallel
        >>>
        >>> model = AnyPytorchModel()
        >>> optimizer = AnyOptimizer(model.parameters(), lr=3e-5)
        >>> pp_wrapper = PipelineParallel(model, ...)

        >>> output = pp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: ParallelContext,
        memory_computation_balance: float = 1.0,
        micro_batch_size: int = 1,
        tracing_inputs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.module = module
        self.micro_batch_size = micro_batch_size
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.local_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)

        self.partitioner = ModelPartitioner(
            module=module,
            process_group=parallel_context.get_group(ParallelMode.PIPELINE),
            tracing_inputs=tracing_inputs,
            memory_computation_balance=memory_computation_balance,
        )
        self.partitioner.partition()
        self.init_model()

    @staticmethod
    def _detect_input_device(args, kwargs):
        devices = []
        for arg in args:
            if torch.is_tensor(arg):
                devices.append(arg.device)

        if len(devices) == 0:
            for key in kwargs:
                if torch.is_tensor(kwargs[key]):
                    devices.append(kwargs[key].device)

        if len(devices) > 0:
            return devices[0]
        else:
            return None

    def module_forward(self, module, module_name, module_forward):
        def new_forward_fns(*args, **kwargs):
            input_device = self._detect_input_device(args, kwargs)

            if input_device is None or input_device == torch.device(
                module.oslo_parallel[ParallelMode.PIPELINE]
            ):
                print(
                    f"SAME m={module_name}, device={input_device}, oslo={module.oslo_parallel[ParallelMode.PIPELINE]}"
                )
                return module_forward(*args, **kwargs)
            else:
                print(
                    f"DIFF m={module_name}, device={input_device}, oslo={module.oslo_parallel[ParallelMode.PIPELINE]}"
                )
                rpc.rpc_async(
                    to=module.oslo_parallel[ParallelMode.PIPELINE],
                    func=module_forward,
                    args=(args, kwargs),
                )

        return new_forward_fns

    def init_model(self):
        for name, module in self.module.named_modules():
            module_forward = module.forward
            module.forward = self.module_forward(module, name, module_forward)

    def split_batch(self, batch):
        return batch.chunk(self.micro_batch_size, dim=0)

    def forward(self, batch):
        if self.local_rank == 0:
            self.module(batch)
