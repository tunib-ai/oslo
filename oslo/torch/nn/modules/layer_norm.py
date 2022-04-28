import torch
import torch.nn as nn
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from torch.nn import Parameter
from oslo.torch.nn.parallel.distributed.tensor_parallel.parallel_2d._ops import (
    layernorm_2d,
    add_bias_2d,
)


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
        self.summa_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
        assert (
            normalized_shape % self.summa_dim == 0
        ), "normalized_shape must be divisible by summa dim."

        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        self.row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        self.col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        self.data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        self.pipeline_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)

        self.tensor_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        self.pipeline_parallel_size = self.parallel_context.get_world_size(ParallelMode.PIPELINE)

        self.partitioned_dim = normalized_shape / self.summa_dim ** 2
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.gamma = Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.beta = Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            E_x = torch.sum(input, dim=-1, keepdim=True)
            dist.all_reduce(E_x, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW))
            E_x /= self.normalized_shape

            Var_x = torch.sum(x * x, dim=-1, keepdim=True)
            dist.all_reduce(Var_x, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW))
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)
        
        output = layernorm_2d(x, E_x, Var_x, self.normalized_shape, ParallelMode.TENSOR_2D_ROW,
                              ParallelMode.TENSOR_2D_COL)
        bias = add_bias_2d(None, self.beta, self.partitioned_dim, self.row_rank, self.col_rank,
                           ParallelMode.TENSOR_2D_ROW, ParallelMode.TENSOR_2D_COL, True, self.data_parallel_rank,
                           self.pipeline_parallel_rank, self.pipeline_parallel_size, self.tensor_parallel_size)
        scale = add_bias_2d(None, self.gamma, self.partitioned_partition, self.row_rank, self.col_rank,
                            ParallelMode.TENSOR_2D_ROW, ParallelMode.TENSOR_2D_COL, True, self.data_parallel_rank,
                            self.pipeline_parallel_rank, self.pipeline_parallel_size, self.tensor_parallel_size)
        output = torch.addcmul(bias, scale, output)
        return output
