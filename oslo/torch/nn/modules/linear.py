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

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
        factory_kwargs = {
            "device": torch.device(torch.cuda.current_device()),
            "dtype": dtype,
        }
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


class ColLinear1D(Linear):
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
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False

        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            out_features % self.world_size == 0
        ), "out_features must be divisible by world_size for ColLinear1D."

        super().__init__(
            in_features=in_features,
            out_features=out_features // self.world_size,
            skip_bias_add=skip_bias_add,
            bias=bias,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_gather_tensor_1d,
            broadcast_tensor_1d,
        )

        input = broadcast_tensor_1d(input, self.parallel_context)
        outputs = F.linear(input, self.weight)

        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                outputs = outputs + self.bias

        if self.gather_output:
            outputs = all_gather_tensor_1d(outputs, -1, self.parallel_context).clone()
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]
            outputs = outputs.contiguous()
        return outputs


class RowLinear1D(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool = False,
        parallel_input: bool = True,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_input = parallel_input
        self.parallel_context = parallel_context
        self.reversed = False

        self.world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        assert (
            in_features % self.world_size == 0
        ), "in_features must be divisible by world_size for RowLinear1D."

        super().__init__(
            in_features=in_features // self.world_size,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, parallel_input={self.parallel_input}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_reduce_tensor_1d,
            scatter_tensor_1d,
        )

        if not self.parallel_input:
            input = scatter_tensor_1d(
                input,
                dim=-1,
                parallel_context=self.parallel_context,
            )

        outputs = F.linear(input, self.weight)
        outputs = all_reduce_tensor_1d(outputs, self.parallel_context)

        if self.bias is not None:
            if self.skip_bias_add:
                return outputs, self.bias
            else:
                return outputs + self.bias

        return outputs.contiguous()


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
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            in_features % self.summa_dim == 0
        ), "in_features must be divisible by summa_dim for Linear2D."
        assert (
            out_features % (self.summa_dim**2) == 0
        ), "out_features must be divisible by summa_dim^2 for Linear2D."

        super().__init__(
            in_features=in_features // self.summa_dim,
            out_features=out_features // self.summa_dim,
            bias=False,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    out_features // (self.summa_dim**2),
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

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

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_2D_COL,
            )
            outputs = all_gather_tensor_2d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_2D_ROW,
            )
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]
            outputs = outputs.contiguous()
            
        return outputs


class Linear2p5D(Linear):
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
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            in_features % self.tesseract_dim == 0
        ), "in_features must be divisible by tesseract_dim for Linear2p5D."
        assert (
            out_features % self.tesseract_dim == 0
        ), "out_features must be divisible by tesseract_dim for Linear2p5D."

        super().__init__(
            in_features=in_features // self.tesseract_dim,
            out_features=out_features // self.tesseract_dim,
            bias=bias,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )

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

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, gather_output={self.gather_output}"
        )

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            Matmul_ABT_2p5D,
            add_bias_2p5d,
            all_gather_tensor_2p5d,
        )

        out_shape = input.shape[:-1] + (self.out_features,)

        outputs = Matmul_ABT_2p5D.apply(
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
                return outputs, bias
            else:
                outputs = add_bias_2p5d(
                    outputs,
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
        if self.gather_output:
            outputs = all_gather_tensor_2p5d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                col_parallel_mode=ParallelMode.TENSOR_2P5D_ROW,
            ).clone()
            outputs = all_gather_tensor_2p5d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                col_parallel_mode=ParallelMode.TENSOR_2P5D_COL,
            ).clone()
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]
            outputs = outputs.contiguous()
            
        return outputs


class Linear3D(Linear):
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
        self.gather_output = gather_output
        self.parallel_context = parallel_context
        self.reversed = False
        self.cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)

        assert (
            in_features % self.cubic_dim == 0
        ), "in_features must be divisible by cubic_dim for Linear3D."
        assert (
            out_features % (self.cubic_dim**2) == 0
        ), "out_features must be divisible by cubic_dim^2 for Linear3D."

        super().__init__(
            in_features=in_features // self.cubic_dim,
            out_features=out_features // (self.cubic_dim**2),
            bias=False,
            dtype=dtype,
            skip_bias_add=skip_bias_add,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    out_features // self.cubic_dim,
                    device=self.weight.device,
                    dtype=dtype,
                )
            )
            self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import (
            Matmul_ABT_3D,
            all_gather_tensor_3d,
        )

        outputs = Matmul_ABT_3D.apply(
            input,
            self.weight,
            self.bias,
            0,
            0,
            0,
            self.parallel_context,
            ParallelMode.TENSOR_3D_INPUT,
            ParallelMode.TENSOR_3D_WEIGHT,
            ParallelMode.TENSOR_3D_OUTPUT,
        )
        if self.gather_output:
            outputs = all_gather_tensor_3d(
                outputs,
                dim=-1,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
            )
            outputs = all_gather_tensor_3d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_INPUT,
            )
            outputs = all_gather_tensor_3d(
                outputs,
                dim=0,
                parallel_context=self.parallel_context,
                parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            )
            if hasattr(self, "orig_num_classes"):
                outputs = outputs[..., : self.orig_num_classes]
        return outputs
