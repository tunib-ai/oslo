import torch
from oslo.pytorch.kernel_fusion.utils.torch_version import higher_than

aten = torch.ops.aten

unrecomputable_ops = [
    aten.mm,
    aten.convolution,
    aten.bmm,
    aten.addmm,
    aten.rand_like,
    aten.randn_like,
    aten.upsample_bilinear2d,

]

if higher_than(1, 11):
    unrecomputable_ops.append(aten.convolution_backward)
    unrecomputable_ops.append(aten.native_dropout)
