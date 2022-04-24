import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelMode


def _broadcast(inputs, parallel_context):
    return inputs


def _all_reduce(inputs, parallel_context):
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
    group = parallel_context.get_group(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return inputs

    dist.all_reduce(tensor=inputs, group=group)
    return inputs


def _all_gather(inputs, parallel_context, dim=-1):
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
    rank = parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
    group = parallel_context.get_group(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return inputs

    tensor_list = [torch.empty_like(inputs) for _ in range(world_size)]
    tensor_list[rank] = inputs
    torch.distributed.all_gather(tensor_list, inputs, group)

    return torch.cat(tensor_list, dim=dim).contiguous()


def _scatter(inputs, parallel_context, dim=-1):
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
    rank = parallel_context.get_local_rank(ParallelMode.TENSOR_1D)

    if world_size == 1:
        return inputs

    tensor_size = inputs.size(dim)
    assert (
        tensor_size % world_size == 0
    ), "tensor_size must be divisible by world size for tensor parallelism"
    split_size_or_sections = tensor_size // world_size

    inputs_list = torch.split(
        inputs, split_size_or_sections=split_size_or_sections, dim=dim
    )
    return inputs_list[rank].contiguous()


class _Broadcast1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return _broadcast(inputs, parallel_context)

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return _all_reduce(grad, parallel_context), None


class _AllReduce1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return _all_reduce(inputs, parallel_context)

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return _broadcast(grad, parallel_context), None


class _AllGather1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return _all_gather(inputs, parallel_context)

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return _scatter(grad, parallel_context), None


class _Scatter1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return _scatter(inputs, parallel_context)

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return _all_gather(grad, parallel_context), None


def broadcast(inputs, parallel_context):
    return _Broadcast1D.apply(inputs, parallel_context)


def all_reduce(inputs, parallel_context):
    return _AllReduce1D.apply(inputs, parallel_context)


def all_gather(inputs, parallel_context):
    return _AllGather1D.apply(inputs, parallel_context)


def scatter(inputs, parallel_context):
    return _Scatter1D.apply(inputs, parallel_context)
