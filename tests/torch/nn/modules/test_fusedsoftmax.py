from oslo.torch.nn import FusedScaleMaskSoftmax
import torch

fused_softmax = FusedScaleMaskSoftmax(
    scale=1.0, use_triang_mask=False, softmax_in_fp32=False
).cuda()
non_fused_softmax = FusedScaleMaskSoftmax(
    scale=1.0, use_triang_mask=False, softmax_in_fp32=True
).cuda()

inputs = torch.randn(16, 16, 16, 16).half().cuda()

fused_output = fused_softmax(inputs)
non_fused_output = non_fused_softmax(inputs).half()
print(f"All close: {torch.allclose(fused_output, non_fused_output, rtol=1e-2)}")
