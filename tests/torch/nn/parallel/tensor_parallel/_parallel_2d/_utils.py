import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelMode


def split_2d(parallel_context, tensor, summa_dim, col_first=True):
    if col_first:
        tensor = tensor.chunk(summa_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        ]
        tensor = tensor.chunk(summa_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        ]
    else:
        tensor = tensor.chunk(summa_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        ]
        tensor = tensor.chunk(summa_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        ]
    return tensor


def split_1d_twice(parallel_context, tensor, summa_dim, dim=-1):
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
    ]
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
    ]
    return tensor


def gather_2d(parallel_context, tensor, summa_dim, col_first=True):
    if col_first:
        tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
        dist.all_gather(
            tensor_list,
            tensor.contiguous(),
            parallel_context.get_group(ParallelMode.TENSOR_2D_COL),
        )
        tensor = torch.cat(tensor_list, dim=0)
        tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
        dist.all_gather(
            tensor_list, tensor, parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
        )
        tensor = torch.cat(tensor_list, dim=-1)
    else:
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


def gather_1d_twice(parallel_context, tensor, summa_dim, dim=-1):
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(ParallelMode.TENSOR_2D_COL),
    )
    tensor = torch.cat(tensor_list, dim=dim)
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list, tensor, parallel_context.get_group(ParallelMode.TENSOR_2D_ROW)
    )
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor
