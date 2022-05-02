from typing import Optional

import torch.distributed as dist
import torch.nn as nn

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.pipeline_parallel._schedulers.pipedream import (
    PipeDreamScheduler,
)
from oslo.torch.nn.parallel.pipeline_parallel._schedulers.scheduler import (
    Scheduler,
)
from oslo.torch.nn.parallel.utils import get_parallel_context


class PipelineParallel(nn.Module):
    """
    Pipeline parallel module

    Args:
        module (nn.Module): PyTorch module object
        process_group (dist.ProcessGroup): process group object
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
        >>> pp_wrapper = PipelineParallel(model)

        >>> output = pp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: Optional[ParallelContext] = None,
        memory_computation_balance: float = 1.0,
        scheduler: Scheduler = PipeDreamScheduler,
    ):
        super().__init__()
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context)
