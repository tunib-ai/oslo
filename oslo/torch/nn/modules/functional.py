from glob import glob
import torch
import torch.nn.functional as F

"""
Autograd Functions
"""
global fused_layer_norm_cuda
fused_layer_norm_cuda = None


# Utils from apex
def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())

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


class _FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output        

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        return grad_input, grad_weight, grad_bias, None, None


class _FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):

        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_norm_backward_affine(
           grad_output.contiguous(), invvar, input_, ctx.normalized_shape, weight_, ctx.eps
        )
        return grad_input, grad_weight, None, None


class _FusedLayerNormAffineMixedDtypesFunction(_FusedLayerNormAffineFunction):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        global fused_layer_norm_cuda
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.layer_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output


class _FusedRMSNormAffineMixedDtypesFunction(_FusedRMSNormAffineFunction):

    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward_affine_mixed_dtypes(
            input_, ctx.normalized_shape, weight_, ctx.eps
        )

        ctx.save_for_backward(input_, weight_, invvar)
        return output


class _FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(input_, ctx.normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, mean, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.backward(
            grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, ctx.eps
        )
        return grad_input, None, None


class _FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_norm_forward(input_, ctx.normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.rms_norm_backward(
            grad_output.contiguous(), invvar, input_, ctx.normalized_shape, ctx.eps
        )
        return grad_input, None, None


def fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineFunction.apply(*args)


def fused_layer_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormFunction.apply(*args)


def mixed_dtype_fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedLayerNormAffineMixedDtypesFunction.apply(*args)


def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineFunction.apply(*args)


def fused_rms_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormFunction.apply(*args)


def mixed_dtype_fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return _FusedRMSNormAffineMixedDtypesFunction.apply(*args)



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
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    # References: `AIB` in https://arxiv.org/abs/2007.00072
    return q_out + q_bias, k_out + k_bias, v_out + v_bias
