import torch
import torch.nn.functional as F


@torch.jit.script
def bias_dropout(x, bias, prob, training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=prob, training=training)


@torch.jit.script
def bias_dropout_train(x, bias, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return bias_dropout(x, bias, prob, True)


@torch.jit.script
def bias_dropout_inference(x, bias, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return bias_dropout(x, bias, prob, False)


@torch.jit.script
def bias_dropout_residual(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=prob, training=training) + residual


@torch.jit.script
def bias_dropout_residual_train(x, bias, residual, prob):
    # type: (Tensor, Tensor,  Tensor, float) -> Tensor
    return bias_dropout_residual(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_residual_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_residual(x, bias, residual, prob, False)
