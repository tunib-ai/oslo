import torch
from torch import Tensor

from oslo.pytorch.kernel_fusion.mem_efficient.decompositions import \
    register_decomposition
from oslo.pytorch.kernel_fusion.utils.torch_version import higher_than

aten = torch.ops.aten

if higher_than(1, 11):

    @register_decomposition(aten.native_dropout_backward)
    def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
        return grad_output * (mask.type_as(grad_output) * scale)

    @register_decomposition(aten.native_dropout)
    def native_dropout_decomposition(input, p, generator=None):
        bool_mask = aten.rand_like(input) < p
        res = bool_mask * input * float(1.0 / p)
        return [res, bool_mask]

    @register_decomposition(aten._reshape_alias)
    def _reshape_alias(x, shape, strides):
        return aten.view(x, shape)
