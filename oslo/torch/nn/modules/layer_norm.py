from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Parameter

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
    add_bias_2d,
    layernorm_2d,
)
from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
    layernorm_2p5d,
    add_bias_2p5d,
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

        self.row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        self.col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
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


class LayerNorm2p5D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        parallel_context: ParallelContext,
        eps: float = 1e-05,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype=None,
    ):
        super().__init__()
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            normalized_shape % self.tesseract_dim == 0
        ), "normalized_shape must be divisible by tessract dim."

        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        self.row_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_ROW
        )
        self.col_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_COL
        )
        self.dep_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_2P5D_DEP
        )
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

        self.partitioned_dim = normalized_shape // self.tesseract_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            E_x = torch.sum(_input, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                E_x, group=self.parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW)
            )
            E_x /= self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(_input * _input, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                Var_x,
                group=self.parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
            )
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x  # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = layernorm_2p5d(
            _input,
            E_x,
            Var_x,
            self.normalized_shape,
            ParallelMode.TENSOR_2P5D_ROW,
            self.parallel_context,
        )
        scale = add_bias_2p5d(
            None,
            self.weight,
            self.partitioned_dim,
            self.tesseract_dim,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            self.parallel_context,
            ParallelMode.TENSOR_2P5D_COL,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )
        if self.bias is not None:
            bias = add_bias_2p5d(
                None,
                self.bias,
                self.partitioned_dim,
                self.tesseract_dim,
                self.row_rank,
                self.col_rank,
                self.dep_rank,
                self.parallel_context,
                ParallelMode.TENSOR_2P5D_COL,
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
