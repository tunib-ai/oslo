from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.distributed.nn._p2p import _P2P
from oslo.torch.distributed.nn._ring_self_attention import _RingQK, _RingAV


def send(
    data: Any,
    src_rank: int,
    dst_rank: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):
    if src_rank == parallel_context.get_local_rank(parallel_mode):
        _P2P().send(
            data=data,
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )


def recv(
    src_rank: int,
    dst_rank: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode = ParallelMode.PIPELINE,
):
    if dst_rank == parallel_context.get_local_rank(parallel_mode):
        return _P2P().recv(
            src_rank=src_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )


def all_gather(
    tensor: Tensor,
    dim: int,
    on_cpu: bool = False,
    async_op: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
) -> Tensor:

    world_size = parallel_context.get_world_size(parallel_mode)

    if world_size == 1:
        out = tensor
        work = None
    else:
        shape = list(tensor.shape)
        shape[0], shape[dim] = shape[dim], shape[0]
        shape[0] *= world_size
        out = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
        temp = list(torch.chunk(out, world_size, dim=0))
        group = (
            parallel_context.get_cpu_group(parallel_mode)
            if on_cpu
            else parallel_context.get_group(parallel_mode)
        )
        work = dist.all_gather(
            tensor_list=temp,
            tensor=tensor.transpose(0, dim).contiguous(),
            group=group,
            async_op=async_op,
        )
        out = torch.transpose(out, 0, dim)
    if async_op:
        return out, work
    else:
        return out


def reduce_scatter(
    tensor: Tensor,
    dim: int,
    op: ReduceOp = ReduceOp.SUM,
    on_cpu: bool = False,
    async_op: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
) -> Tensor:
    world_size = parallel_context.get_world_size(parallel_mode)

    if world_size == 1:
        out = tensor
        work = None
    else:
        assert (
            tensor.size(dim) % world_size == 0
        ), "tensor_size must be divisible by world size for tensor parallelism"
        temp = list(
            map(lambda x: x.contiguous(), torch.chunk(tensor, world_size, dim=dim))
        )
        out = torch.empty(temp[0].shape, dtype=tensor.dtype, device=tensor.device)
        group = (
            parallel_context.get_cpu_group(parallel_mode)
            if on_cpu
            else parallel_context.get_group(parallel_mode)
        )
        work = dist.reduce_scatter(
            output=out, input_list=temp, op=op, group=group, async_op=async_op
        )
    if async_op:
        return out, work
    else:
        return out


def all_reduce(
    tensor: Tensor,
    op: ReduceOp = ReduceOp.SUM,
    on_cpu: bool = False,
    async_op: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
) -> Tensor:
    world_size = parallel_context.get_world_size(parallel_mode)
    if world_size == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        group = (
            parallel_context.get_cpu_group(parallel_mode)
            if on_cpu
            else parallel_context.get_group(parallel_mode)
        )
        work = dist.all_reduce(out, op=op, group=group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def broadcast(
    tensor: Tensor,
    src: int,
    on_cpu: bool = False,
    async_op: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
):
    world_size = parallel_context.get_world_size(parallel_mode)

    if world_size == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        group = (
            parallel_context.get_cpu_group(parallel_mode)
            if on_cpu
            else parallel_context.get_group(parallel_mode)
        )
        work = dist.broadcast(out, src=src, group=group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def reduce(
    tensor: Tensor,
    dst: int,
    op: ReduceOp = ReduceOp.SUM,
    on_cpu: bool = False,
    async_op: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
):
    world_size = parallel_context.get_world_size(parallel_mode)
    if world_size == 1:
        out = tensor
        work = None
    else:
        out = tensor.contiguous()
        group = (
            parallel_context.get_cpu_group(parallel_mode)
            if on_cpu
            else parallel_context.get_group(parallel_mode)
        )
        work = dist.reduce(out, dst=dst, op=op, group=group, async_op=async_op)
    if async_op:
        return out, work
    else:
        return out


def scatter(
    tensor: Tensor,
    dim: int,
    parallel_context: Optional[ParallelContext] = None,
    parallel_mode: Optional[ParallelMode] = None,
):
    world_size = parallel_context.get_world_size(parallel_mode)
    rank = parallel_context.get_local_rank(parallel_mode)

    if world_size == 1:
        return tensor

    assert (
        tensor.size(dim) % world_size == 0
    ), "tensor_size must be divisible by world size for tensor parallelism"

    tensor_list = torch.chunk(tensor, world_size, dim=dim)
    return tensor_list[rank].contiguous()


def scatter_object_list(
    scatter_object_output_list, scatter_object_input_list, src=0, group=None
):
    r"""Modified from `torch.distributed.scatter_object_list <https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#scatter_object_list>` to fix issues"""
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    if (
        not isinstance(scatter_object_output_list, list)
        or len(scatter_object_output_list) < 1
    ):
        raise RuntimeError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    # set tensor device to cuda if backend is nccl
    device = (
        torch.cuda.current_device()
        if dist.get_backend(group) == "nccl"
        else torch.device("cpu")
    )

    my_rank = dist.get_rank()  # use global rank
    if my_rank == src:
        tensor_list, tensor_sizes = zip(
            *[
                dist.distributed_c10d._object_to_tensor(obj)
                for obj in scatter_object_input_list
            ]
        )
        tensor_list = list(map(lambda x: x.to(device), tensor_list))
        tensor_sizes = list(map(lambda x: x.to(device), tensor_sizes))

    # Src rank broadcasts the maximum tensor size. This is because all ranks are
    # expected to call into scatter() with equal-sized tensors.
    if my_rank == src:
        max_tensor_size = max(tensor_sizes)
        for tensor in tensor_list:
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long).to(device)

    dist.broadcast(max_tensor_size, src=src, group=group)

    # Scatter actual serialized objects
    output_tensor = torch.empty(max_tensor_size.item(), dtype=torch.uint8).to(device)
    dist.scatter(
        output_tensor,
        scatter_list=None if my_rank != src else tensor_list,
        src=src,
        group=group,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long).to(device)
    dist.scatter(
        obj_tensor_size,
        scatter_list=None if my_rank != src else tensor_sizes,
        src=src,
        group=group,
    )

    output_tensor, obj_tensor_size = output_tensor.cpu(), obj_tensor_size.cpu()
    # Deserialize back to object
    scatter_object_output_list[0] = dist.distributed_c10d._tensor_to_object(
        output_tensor, obj_tensor_size
    )


def ring_qk(sub_q: Tensor, sub_k: Tensor, parallel_context: ParallelContext):
    return _RingQK.apply(sub_q, sub_k, parallel_context)


def ring_av(sub_attn: Tensor, sub_v: Tensor, parallel_context: ParallelContext):
    return _RingAV.apply(sub_attn, sub_v, parallel_context)
