import torch
import torch.nn.functional as F

"""
Autograd Functions
"""


@torch.jit.script
def _fused_gelu_fwb(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


@torch.jit.script
def _fused_bias_gelu_fwb(y, bias):
    x = y + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def _fused_bias_gelu_bwd(g, y, bias):
    x = y + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class _FusedGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: GeLU
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return _fused_gelu_fwb(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = _fused_gelu_bwd(grad_output, input)
        return tmp, tmp


class _FusedBiasGeLUFunction(torch.autograd.Function):
    """
    Kernel fusion function: Bias + GeLU
    """

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _fused_bias_gelu_fwb(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = _fused_bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


"""
User Functions
"""


def fused_gelu(x):
    return _FusedGeLUFunction.apply(x)


def fused_bias_gelu(x, bias):
    return _FusedBiasGeLUFunction.apply(x, bias)


@torch.jit.script
def fused_bias_dropout(x, bias, p, training, inplace):
    # type: (Tensor, Tensor, float, bool, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training, inplace=inplace)


@torch.jit.script
def fused_bias_dropout_residual(x, bias, residual, p, training, inplace):
    # type: (Tensor, Tensor, Tensor, float, bool, bool) -> Tensor
    return F.dropout(x + bias, p=p, training=training, inplace=inplace) + residual


@torch.jit.script
def fused_attention_input_bias(q_out, k_out, v_out, q_bias, k_bias, v_bias):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # References: `AIB` in https://arxiv.org/abs/2007.00072
    return q_out + q_bias, k_out + k_bias, v_out + v_bias
