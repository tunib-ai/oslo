import torch

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import Linear2D


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2d",
)

layer_2d = Linear2D(4, 4, parallel_context)
layer_2d(torch.randn(2, 2).cuda())
