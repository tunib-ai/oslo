import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Parameter

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
    add_bias_2d,
    layernorm_2d,
)


class LayerNorm2D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        parallel_context: ParallelContext,
        eps: float = 1e-05,
        bias=True,
        dtype=None,
    ):
        super().__init__()
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            normalized_shape % self.summa_dim == 0
        ), "normalized_shape must be divisible by summa dim."

        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        self.row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        self.col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        self.data_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.DATA
        )
        self.pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )

        self.tensor_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.TENSOR
        )
        self.pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        self.partitioned_dim = normalized_shape // (self.summa_dim**2)

        factory_kwargs = {
            "device": torch.device(torch.cuda.current_device()),
            "dtype": dtype,
        }
        self.weight = Parameter(torch.ones(self.partitioned_dim, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.partitioned_dim, **factory_kwargs))
        else:
            self.bias = None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            E_i = torch.sum(input_, dim=-1, keepdim=True)
            dist.all_reduce(
                E_i, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
            )
            E_i /= self.normalized_shape

            Var_i = torch.sum(input_ * input_, dim=-1, keepdim=True)
            dist.all_reduce(
                Var_i, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
            )
            Var_i /= self.normalized_shape

            Var_i = Var_i - E_i * E_i
            Var_i = 1.0 / torch.sqrt(Var_i + self.variance_epsilon)

        output = layernorm_2d(
            input_,
            E_i,
            Var_i,
            self.normalized_shape,
            self.parallel_context,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
        )
        scale = add_bias_2d(
            None,
            self.weight,
            self.partitioned_dim,
            self.row_rank,
            self.col_rank,
            self.parallel_context,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )
        if self.bias is not None:
            bias = add_bias_2d(
                None,
                self.bias,
                self.partitioned_dim,
                self.row_rank,
                self.col_rank,
                self.parallel_context,
                ParallelMode.TENSOR_2D_ROW,
                ParallelMode.TENSOR_2D_COL,
                True,
                self.data_parallel_rank,
                self.pipeline_parallel_rank,
                self.pipeline_parallel_size,
                self.tensor_parallel_size,
            )
            output = torch.addcmul(bias, scale, output)
        else:
            output = torch.mul(scale, output)
        return output
