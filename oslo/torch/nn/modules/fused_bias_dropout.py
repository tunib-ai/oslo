import torch
import torch.nn.functional as F


@torch.jit.script
def fused_bias_dropout(x, bias, p, training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training)


@torch.jit.script
def fused_bias_dropout_residual(x, bias, residual, p, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training) + residual
