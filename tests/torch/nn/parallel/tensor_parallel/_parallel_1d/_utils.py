import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelMode


def split_1d(parallel_context, tensor, world_size, dim):
    tensor = tensor.chunk(world_size, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
    ]
    return tensor

def gather_1d(parallel_context, tensor, world_size, dim):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous(), parallel_context.get_group(ParallelMode.TENSOR_1D))
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor
