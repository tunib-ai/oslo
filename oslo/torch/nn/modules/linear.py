from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.parameter import UninitializedParameter

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.lazy import LazyModuleMixin


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
    ):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )
        self.skip_bias_add = skip_bias_add

    def forward(
        self, input: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not self.skip_bias_add:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight), self.bias


class LazyLinear(LazyModuleMixin, Linear):
    """
    Lazy initialized linear layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of parameters are copied to the CPU RAM, which causes the CPU RAM to run out.
    Therefore, after creating uninitialized parameters and re-adjusting them to a suitable size,
    you can initialize only the necessary parameters to a suitable GPU immediately.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.

    Notes:
        This is different from torch.nn.LazyLinear in terms of
        ``in_features`` can be input by user at the creation time.

    Examples:
        >>> from oslo.torch.nn import LazyLinear

        >>> layer = LazyLinear(2, 4)
        >>> print(layer.weight)
        <UninitializedParameter>

        >>> layer.initialize_parameters()
        >>> print(layer.weight)
        Parameter containing:
        tensor([[-0.7025,  0.5608],
                [-0.2529, -0.2636],
                [-0.5755, -0.2422],
                [ 0.4704,  0.6281]], requires_grad=True)
    """

    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
    ) -> None:
        factory_kwargs = {"device": torch.device(torch.cuda.current_device()), "dtype": dtype}
        super().__init__(0, 0, False)
        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        self.weight = UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.cls_to_become is not None:
            self.__class__ = self.cls_to_become


class ColumnParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
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
        self, input: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # to avoid circular import
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_gather_1d,
            broadcast_1d,
        )

        input = broadcast_1d(input, self.parallel_context)
        outputs = F.linear(input, self.weight)

        if self.gather_output:
            outputs = all_gather_1d(outputs, -1, self.parallel_context).clone()

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
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_input: bool = True,
        skip_bias_add: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_input = parallel_input
        self.parallel_context = parallel_context
        self.reversed = False

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            in_features % world_size == 0
        ), "in_features must be divisible by world_size for tensor parallelism."

        super().__init__(
            in_features=in_features // world_size,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            device=torch.device(torch.cuda.current_device()),
            skip_bias_add=skip_bias_add,
        )

    def forward(
        self, input: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # to avoid circular import
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_reduce_1d,
            scatter_1d,
        )
        if self.parallel_input:
            assert input.shape[-1] == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input.shape, self.weight.shape, self.weight.shape[-1])
            input = input
        else:
            assert input.shape[-1] // self.parallel_context.tensor_parallel_size == self.weight.shape[-1], \
            'Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.'.format(
                input.shape, self.weight.shape, self.weight.shape[-1] * self.parallel_context.tensor_parallel_size)
            input = scatter_1d(input, -1, self.parallel_context)

        outputs = F.linear(input, self.weight)
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
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        parallel_context: Optional[ParallelContext] = None,
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
            in_features=in_features // self.summa_dim,
            out_features=out_features // self.summa_dim,
            bias=False,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.out_features // (self.summa_dim ** 2),
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

    def forward(
        self, input: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
            out_shape,
            self.row_rank,
            self.col_rank,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2D_ROW,
            ParallelMode.TENSOR_2D_COL,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2d(
                    None,
                    self.bias,
                    self.out_features,
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
                return outputs, bias
            else:
                outputs = add_bias_2d(
                    outputs,
                    self.bias,
                    self.out_features,
                    self.row_rank,
                    self.col_rank,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2D_ROW,
                    ParallelMode.TENSOR_2D_COL,
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
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        parallel_context: Optional[ParallelContext] = None,
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

    def forward(self, input: Tensor) -> Tensor:
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            Matmul_ABT_2p5D,
            add_bias_2p5d,
        )

        out_shape = input.shape[:-1] + (self.out_features,)

        output = Matmul_ABT_2p5D.apply(
            input,
            self.weight,
            self.tesseract_dim,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
            self.parallel_context,
            ParallelMode.TENSOR_2P5D_ROW,
            ParallelMode.TENSOR_2P5D_COL,
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
                    True,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2P5D_COL,
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
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size,
                    self.parallel_context,
                    ParallelMode.TENSOR_2P5D_COL,
                )
                return output
        else:
            return output


class Linear3D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)

        assert self.cubic_dim > 0, "CUBIC_DIM must be greater than zero"
        assert (
            in_features % self.cubic_dim == 0
        ), "in_features must be divisible by cubic dim."
        assert (
            out_features % (self.cubic_dim ** 2) == 0
        ), "out_features must be divisible by (cubic dim)^2."

        self.input_parallel_mode = ParallelMode.TENSOR_3D_INPUT
        self.weight_parallel_mode = ParallelMode.TENSOR_3D_WEIGHT
        self.output_parallel_mode = ParallelMode.TENSOR_3D_OUTPUT

        super().__init__(
            in_features=in_features // self.cubic_dim,
            out_features=out_features // (self.cubic_dim ** 2),
            bias=False,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.out_features // self.cubic_dim,
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import linear_3d
        return linear_3d(
            input, 
            self.weight, 
            self.bias, 
            parallel_context=self.parallel_context,
            input_parallel_mode=self.input_parallel_mode, 
            weight_parallel_mode=self.weight_parallel_mode,
            output_parallel_mode=self.output_parallel_mode,
        )