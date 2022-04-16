from oslo.torch._C import SoftmaxBinder
from oslo.torch import nn

# CUDA = SoftmaxBinder().bind()
# print(CUDA.scaled_masked_softmax_forward)
# print(CUDA.scaled_masked_softmax_backward)
# print(CUDA.get_batch_per_block)
# print(CUDA.scaled_upper_triang_masked_softmax_forward)
# print(CUDA.scaled_upper_triang_masked_softmax_backward)

# print(nn.Conv1D)
print(nn.FusedScaleMaskSoftmax)