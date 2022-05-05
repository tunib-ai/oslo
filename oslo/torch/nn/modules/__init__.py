from oslo.torch.jit._utils import _set_jit_fusion_options
from oslo.torch._C import SoftmaxBinder

_SOFTMAX_KERNEL = None


def get_softmax_kernel():
    global _SOFTMAX_KERNEL

    try:
        if _SOFTMAX_KERNEL is None:
            _set_jit_fusion_options()
            _SOFTMAX_KERNEL = SoftmaxBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _SOFTMAX_KERNEL
