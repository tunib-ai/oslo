import torch

from oslo.torch.distributed import ParallelContext, ParallelMode

gpc = ParallelContext.from_torch(
    data_parallel_size=2,
    pipeline_parallel_size=2,
    tensor_parallel_size=2,
    tensor_parallel_mode="1d",
)

if torch.distributed.get_rank() == 0:
    print(gpc.get_world_size(ParallelMode.TENSOR))
    print(gpc.get_world_size(ParallelMode.DATA))
    print(gpc.get_world_size(ParallelMode.PIPELINE))
