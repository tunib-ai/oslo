from oslo.torch.nn import get_softmax_kernel

_SOFTMAX_KERNEL = None


def test_get_softmax_kernels():
    CUDA = get_softmax_kernel()
    CUDA.scaled_masked_softmax_forward
    CUDA.scaled_masked_softmax_backward
    CUDA.get_batch_per_block
    CUDA.scaled_upper_triang_masked_softmax_forward
    CUDA.scaled_upper_triang_masked_softmax_backward
    print("test_get_kernels passed")


test_get_softmax_kernels()
