from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from oslo.torch.distributed import ParallelContext, ParallelMode


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
    ):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.skip_bias_add = skip_bias_add

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.skip_bias_add:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight), self.bias




class ColumnParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        skip_bias_add: bool = False,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
    ):
        self.parallel_context = parallel_context
        self.gather_output = gather_output
        self.reversed = False

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            out_features % world_size == 0
        ), "out_features must be divisible by world_size for tensor parallelism."

        super().__init__(
            in_features=in_features,
            out_features=out_features // world_size,
            skip_bias_add=skip_bias_add,
            bias=bias,
            dtype=dtype,
            device=torch.device(torch.cuda.current_device()),
        )

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # to avoid circular import
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_gather_1d,
            broadcast_1d,
        )

        input = broadcast_1d(input, self.parallel_context)
        outputs = torch.matmul(input, self.weight.t())

        if self.gather_output:
            outputs = all_gather_1d(outputs, self.parallel_context).clone()

        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                return outputs + self.bias

        return outputs


class RowParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
    ):
        self.parallel_context = parallel_context
        self.reversed = False

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            in_features % world_size == 0
        ), "in_features must be divisible by world_size for tensor parallelism."

        super().__init__(
            in_features=in_features // world_size,
            out_features=out_features,
            skip_bias_add=skip_bias_add,
            bias=bias,
            dtype=dtype,
            device=torch.device(torch.cuda.current_device()),
        )

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # to avoid circular import
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_reduce_1d,
        )

        outputs = torch.matmul(input, self.weight.t())
        outputs = all_reduce_1d(outputs, self.parallel_context)

        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                return outputs + self.bias

        return outputs


class Linear2D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
    ):
        self.parallel_context = parallel_context
        self.gather_output = gather_output
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            in_features % self.summa_dim == 0
        ), "in_features must be divisible by summa dim."
        assert (
            out_features % (self.summa_dim**2) == 0
        ), "out_features must be divisible by summa dim^2."

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

        super().__init__(
            in_features=int(in_features // self.summa_dim),
            out_features=int(out_features // self.summa_dim),
            bias=False,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.out_features // self.summa_dim,
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
        super().reset_parameters()

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
            Matmul_ABT_2D,
            add_bias_2d,
            all_gather_tensor_2d,
        )

        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = input.shape[:-1] + (self.out_features,)
        outputs = Matmul_ABT_2D.apply(
            input,
            self.weight,
            self.summa_dim,
            self.parallel_context,
            out_shape,
            self.row_rank,
            self.col_rank,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2d(
                    None,
                    self.bias,
                    self.out_features,
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
                return outputs, bias
            else:
                outputs = add_bias_2d(
                    outputs,
                    self.bias,
                    self.out_features,
                    self.row_rank,
                    self.col_rank,
                    self.parallel_context,
                    ParallelMode.TENSOR_2D_ROW,
                    ParallelMode.TENSOR_2D_COL,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                )
        if self.gather_output:
            outputs = all_gather_tensor_2d(
                outputs,
                dim=-1,
                parallel_mode=ParallelMode.TENSOR_2D_COL,
                parallel_context=self.parallel_context,
            ).clone()
        return outputs


class Linear2p5D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel_context: ParallelContext,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        skip_bias_add: bool = False,
    ):
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert self.tesseract_dim > 0, "TESSERACT_DIM must be larger than zero"
        assert (
            in_features % self.tesseract_dim == 0
        ), "in_features must be divisible by tesseract dim."
        assert (
            out_features % self.tesseract_dim == 0
        ), "out_features must be divisible by tesseract dim."

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

        super().__init__(
            in_features=int(in_features // self.tesseract_dim),
            out_features=int(out_features // self.tesseract_dim),
            bias=bias,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            all_reduce,
            Matmul_AB_2p5D,
            Matmul_ABT_2p5D,
            add_bias_2p5d,
        )

        out_shape = input.shape[:-1] + (self.out_features,)

        output = Matmul_ABT_2p5D.apply(
            input,
            self.weight,
            self.tesseract_dim,
            self.parallel_context,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            ParallelMode.TENSOR_2P5D_ROW,
            ParallelMode.TENSOR_2P5D_COL,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2p5d(
                    None,
                    self.bias,
                    self.out_features,
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
                return output, bias
            else:
                output = add_bias_2p5d(
                    output,
                    self.bias,
                    self.out_features,
                    self.tesseract_dim,
                    self.row_rank,
                    self.col_rank,
                    self.dep_rank,
                    self.parallel_context,
                    ParallelMode.TENSOR_2P5D_COL,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                )
                return output
        else:
            return output
