import torch.nn as nn
import torch.distributed as dist

from oslo.torch.nn.parallel.distributed.pipeline_parallel._schedulers.pipedream import (
    PipeDreamScheduler,
)
from oslo.torch.nn.parallel.distributed.pipeline_parallel._schedulers.scheduler import (
    Scheduler,
)


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
        3.

    """

    def __init__(
        self,
        module,
        process_group: dist.ProcessGroup,
        memory_computation_balance: float = 1.0,
        scheduler: Scheduler = PipeDreamScheduler,
    ):
        super().__init__()
