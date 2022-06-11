import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelMode


def split_batch_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor

def split_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor

def split_embedding_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=-1)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor

def split_layernorm_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor

def split_bias_2d(tensor, summa_dim, parallel_context):
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def gather_2d(tensor, summa_dim, parallel_context):
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list, tensor, parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
    )
    tensor = torch.cat(tensor_list, dim=-1)
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_2D_COL),
    )
    tensor = torch.cat(tensor_list, dim=0)
    return tensor
