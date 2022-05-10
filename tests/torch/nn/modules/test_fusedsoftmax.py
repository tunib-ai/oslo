from oslo.torch.nn import FusedScaleMaskSoftmax
import torch

softmax = FusedScaleMaskSoftmax(scale=1.1, use_triang_mask=True).cuda()
inputs = torch.randn(10, 10, 10, 10).half().cuda()

out = softmax(inputs)
print(out)


