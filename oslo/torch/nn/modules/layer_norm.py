from typing import Optional

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Parameter

from oslo.torch.distributed import ParallelContext, ParallelMode


class LayerNorm2D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
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
        self.eps = eps

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

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
            add_bias_2d,
            layernorm_2d,
        )

        with torch.no_grad():
            E_i = torch.sum(input, dim=-1, keepdim=True)
            dist.all_reduce(
                E_i, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
            )
            E_i /= self.normalized_shape

            Var_i = torch.sum(input * input, dim=-1, keepdim=True)
            dist.all_reduce(
                Var_i, group=self.parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
            )
            Var_i /= self.normalized_shape

            Var_i = Var_i - E_i * E_i
            Var_i = 1.0 / torch.sqrt(Var_i + self.eps)

        output = layernorm_2d(
            input,
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
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
        )
        if self.bias is not None:
            bias = add_bias_2d(
                None,
                self.bias,
                self.partitioned_dim,
                self.row_rank,
                self.col_rank,
                True,
                self.data_parallel_rank,
                self.pipeline_parallel_rank,
                self.pipeline_parallel_size,
                self.tensor_parallel_size,
                self.parallel_context,
                ParallelMode.TENSOR_2D_ROW,
                ParallelMode.TENSOR_2D_COL,
            )
            output = torch.addcmul(bias, scale, output)
        else:
            output = torch.mul(scale, output)
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, partitioned_dim={self.partitioned_dim}, "
            f"eps={self.eps}, elementwise_affine={self.elementwise_affine}"
        )


class LayerNorm2p5D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
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

        factory_kwargs = {
            "device": torch.device(torch.cuda.current_device()),
            "dtype": dtype,
        }

        self.weight = Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))

    def forward(self, input: Tensor) -> torch.Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            layernorm_2p5d,
            add_bias_2p5d,
        )

        with torch.no_grad():
            E_x = torch.sum(input, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                E_x, group=self.parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW)
            )
            E_x /= self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(input * input, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                Var_x,
                group=self.parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
            )
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x  # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = layernorm_2p5d(
            input,
            E_x,
            Var_x,
            self.normalized_shape,
            self.parallel_context,
            ParallelMode.TENSOR_2P5D_ROW,
        )
        scale = add_bias_2p5d(
            None,
            self.weight,
            self.partitioned_dim,
            self.tesseract_dim,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2P5D_COL,
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
                True,
                self.data_parallel_rank,
                self.pipeline_parallel_rank,
                self.pipeline_parallel_size,
                self.tensor_parallel_size,
                self.parallel_context,
                ParallelMode.TENSOR_2P5D_COL,
            )
            output = torch.addcmul(bias, scale, output)
        else:
            output = torch.mul(scale, output)
        return output


class LayerNorm3D(nn.Module):
    def __init__(
        self, 
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):

        super().__init__()
        self.parallel_context = parallel_context
        self.cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
        self.normalized_shape = normalized_shape
        self.partitioned_dim = normalized_shape // self.cubic_dim

        self.input_parallel_mode = ParallelMode.TENSOR_3D_INPUT
        self.weight_parallel_mode = ParallelMode.TENSOR_3D_WEIGHT
        self.output_parallel_mode = ParallelMode.TENSOR_3D_OUTPUT

        factory_kwargs = {
            "device": torch.device(torch.cuda.current_device()),
            "dtype": dtype,
        }

        self.weight = Parameter(torch.ones(self.normalized_shape_per_partition, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.normalized_shape_per_partition, **factory_kwargs))
        else:
            self.bias = None
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import (
            layernorm_3d,
        )
        return layernorm_3d(
            input, 
            self.weight, 
            self.bias, 
            self.normalized_shape, 
            self.eps,
            self.parallel_context,
            self.input_parallel_mode, 
            self.weight_parallel_mode, 
            self.output_parallel_mode,
        )

