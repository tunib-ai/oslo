import torch.nn as nn

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext
from oslo.torch.nn.parallel.utils import ParallelWrapper


class ExpertParallel(ParallelWrapper):
    def __init__(
        self,
        model: nn.Module,
        parallel_context: ParallelContext,
        use_kernel_optim=True,
    ):
        super().__init__()
        self.ep_context = ExpertParallelContext(parallel_context, use_kernel_optim)
        self.ep_context.setup(parallel_context.seed)
