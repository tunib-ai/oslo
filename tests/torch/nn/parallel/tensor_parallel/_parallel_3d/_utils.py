import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelMode


def split_batch_3d(parallel_context, tensor, cubic_dim):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    return tensor


def split_input_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor

def split_weight_3d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
    ]
    return tensor

def split_1d(tensor, cubic_dim, dim, parallel_context, parallel_mode):
    tensor = tensor.chunk(cubic_dim, dim=dim)[
        parallel_context.get_local_rank(parallel_mode)
    ]
    return tensor

def split_bias_1d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
    ]
    return tensor

def split_embedding_1d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor

def split_layernorm_1d(tensor, cubic_dim, parallel_context):
    tensor = tensor.chunk(cubic_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
    ]
    return tensor

def gather_output_3d(tensor, cubic_dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_OUTPUT),
    )
    tensor = torch.cat(tensor_list, dim=-1)
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_INPUT),
    )
    tensor = torch.cat(tensor_list, dim=0)
    tensor_list = [torch.zeros_like(tensor) for _ in range(cubic_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_3D_WEIGHT),
    )
    tensor = torch.cat(tensor_list, dim=0)
    return tensor