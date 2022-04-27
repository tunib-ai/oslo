import torch
import torch.nn as nn
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from torch.nn import Parameter


class LayerNorm2D(nn.Module):
    def __init__(
        self, 
        normalized_shape: int, 
        parallel_context: ParallelContext, 
        eps: float=1e-05, 
        dtype=None
    ):
        super().__init__()
        self.parallel_context = parallel_context
        self.summa_dim = math.sqrt(self.parallel_context.get_world_size(ParallelMode.TENSOR))
        assert (
            normalized_shape % self.summa_dim == 0
        ), "normalized_shape must be divisible by summa dim."

        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        self.row_rank = self.parallel_context.get_local_rank(ParallelMode.PARALLEL_2D_COL)
        self.col_rank = self.parallel_context.get_local_rank(ParallelMode.PARALLEL_2D_ROW)

        self.partitioned_dim = normalized_shape / self.summa_dim ** 2
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.gamma = Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.beta = Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            E_x = torch.sum(input, dim=-1, keepdim=True)
            dist.all_reduce(E_x, group=self.parallel_context.get_group(ParallelMode.PARALLEL_2D_ROW))
            E_x /= self.normalized_shape

            Var_x = torch.sum(x * x, dim=-1, keepdim=True)
            dist.all_reduce(Var_x, group=self.parallel_context.get_group(ParallelMode.PARALLEL_2D_ROW))
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)
            