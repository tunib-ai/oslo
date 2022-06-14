from typing import Optional
import numbers

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
from torch.nn import functional as F

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch import nn as onn


# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        partitioned_dim: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.partitioned_dim = partitioned_dim
        self.eps = eps
        self.elementwise_affine = True

        factory_kwargs = {
            "device": torch.device(torch.cuda.current_device()),
            "dtype": dtype,
        }
        self.weight = Parameter(torch.ones(self.partitioned_dim, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.partitioned_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, partitioned_dim={self.partitioned_dim}, "
            f"eps={self.eps}, elementwise_affine={self.elementwise_affine}"
        )

    def forward(self, input: Tensor) -> Tensor:
        normalized_shape = (
            (self.normalized_shape,)
            if isinstance(self.normalized_shape, int)
            else self.normalized_shape
        )
        return F.layer_norm(input, normalized_shape, self.weight, self.bias, self.eps)


class LayerNorm1D(LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        super().__init__(
            normalized_shape=normalized_shape,
            partitioned_dim=normalized_shape,
            eps=eps,
            bias=bias,
            dtype=dtype,
        )


class LayerNorm2D(LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        assert (
            normalized_shape % (self.summa_dim**2) == 0
        ), "normalized_shape must be divisible by summa_dim^2 for LayerNorm2D."

        super().__init__(
            normalized_shape=normalized_shape,
            partitioned_dim=normalized_shape // (self.summa_dim**2),
            eps=eps,
            bias=bias,
            dtype=dtype,
        )

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

        outputs = layernorm_2d(
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
            outputs = torch.addcmul(bias, scale, outputs)
        else:
            outputs = torch.mul(scale, outputs)
        return outputs


class LayerNorm2p5D(LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        assert (
            normalized_shape % self.tesseract_dim == 0
        ), "normalized_shape must be divisible by tesseract_dim for LayerNorm2p5D."

        super().__init__(
            normalized_shape=normalized_shape,
            partitioned_dim=normalized_shape // self.tesseract_dim,
            eps=eps,
            bias=bias,
            dtype=dtype,
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
            Var_x = 1.0 / torch.sqrt(Var_x + self.eps)

        outputs = layernorm_2p5d(
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
            outputs = torch.addcmul(bias, scale, outputs)
        else:
            outputs = torch.mul(scale, outputs)
        return outputs


class LayerNorm3D(LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-05,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
        assert (
            normalized_shape % self.cubic_dim == 0
        ), "normalized_shape must be divisible by cubic_dim for LayerNorm3D."

        super().__init__(
            normalized_shape=normalized_shape,
            partitioned_dim=normalized_shape // self.cubic_dim,
            eps=eps,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import (
            layernorm_3d,
        )
        outputs = layernorm_3d(
            input,
            self.weight,
            self.bias,
            self.normalized_shape,
            self.eps,
            parallel_context=self.parallel_context,
            input_parallel_mode=ParallelMode.TENSOR_3D_INPUT,
            weight_parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
            output_parallel_mode=ParallelMode.TENSOR_3D_OUTPUT,
        )
        return outputs


class FusedLayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedLayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedLayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedLayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

    def forward(self, input):
        if not input.is_cuda:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if self.elementwise_affine:
            return onn.fused_layer_norm_affine(
                input, self.weight, self.bias, self.normalized_shape, self.eps
            )
        else:
            return onn.fused_layer_norm(input, self.normalized_shape, self.eps)


class FusedRMSNorm(nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        if not input.is_cuda:
            return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

        if self.elementwise_affine:
            return onn.fused_rms_norm_affine(
                input, self.weight, self.normalized_shape, self.eps
            )
        else:
            return onn.fused_rms_norm(input, self.normalized_shape, self.eps)


# NOTE (mkozuki): Why "mixed"?
# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
class MixedFusedLayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        if "elementwise_affine" in kwargs:
            import warnings

            warnings.warn(
                "MixedFusedLayerNorm does not support `elementwise_affine` argument"
            )
            elementwise_affine = kwargs.pop("elementwise_affine")
            if not elementwise_affine:
                raise RuntimeError(
                    "MixedFusedLayerNorm does not support `elementwise_affine = False`"
                )

        super().__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=True
        )

    def forward(self, input: torch.Tensor):
        # NOTE (mkozuki): CPU path is here mainly for unittest sake.
        if not input.is_cuda:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        return onn.mixed_dtype_fused_layer_norm_affine(
            input, self.weight, self.bias, self.normalized_shape, self.eps
        )


# MixedFusedLayerNorm differs from FusedLayerNorm in that this layer norm uses parameter's dtype
# as output tensor's dtype while FusedLayerNorm uses input tensor's dtype for output tensor's dtype.
# See: `layer_norm_affine` and `layer_norm_affine_mixed_dtypes` in "csrc/layer_norm_cuda.cpp"
class MixedFusedRMSNorm(FusedRMSNorm):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        if "elementwise_affine" in kwargs:
            import warnings

            warnings.warn(
                "MixedFusedRMSNorm does not support `elementwise_affine` argument"
            )
            elementwise_affine = kwargs.pop("elementwise_affine")
            if not elementwise_affine:
                raise RuntimeError(
                    "MixedFusedRMSNorm does not support `elementwise_affine = False`"
                )

        super().__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=True
        )

    def forward(self, input: torch.Tensor):
        # NOTE (mkozuki): CPU path is here mainly for unittest sake.
        # TODO Manual RMS Norm Implementation Here
        if not input.is_cuda:
            return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)
        return onn.mixed_dtype_fused_rms_norm_affine(
            input, self.weight, self.normalized_shape, self.eps
        )
