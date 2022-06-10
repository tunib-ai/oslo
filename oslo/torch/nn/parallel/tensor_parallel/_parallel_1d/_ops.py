from typing import Any

import torch
from torch import Tensor

from oslo.torch.distributed import ParallelMode, ParallelContext
from oslo.torch.distributed.nn.functional import all_gather, all_reduce, scatter


class _Broadcast1D(torch.autograd.Function):
    def forward(ctx: Any, inputs: Tensor, parallel_context: ParallelContext):
        if ctx:
            ctx.parallel_context = parallel_context
        return inputs

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return (
            all_reduce(
                grad,
                on_cpu=str(grad.device) == "cpu",
                async_op=False,
                parallel_context=parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
        )


class _AllReduce1D(torch.autograd.Function):
    def forward(ctx: Any, inputs: Tensor, parallel_context: ParallelContext):
        return all_reduce(
            inputs,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    def backward(ctx, grad):
        return grad, None


class _AllGather1D(torch.autograd.Function):
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_context: ParallelContext):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
        return all_gather(
            inputs,
            dim=dim,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_1D,
        )

    def backward(ctx: Any, grad: Tensor):
        return (
            scatter(
                grad,
                dim=ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
            None,
        )


class _Scatter1D(torch.autograd.Function):
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_context: ParallelContext):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
        return (
            scatter(
                inputs,
                dim=dim,
                parallel_context=parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
        )

    def backward(ctx, grad):
        return (
            all_gather(
                grad,
                dim=ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ParallelMode.TENSOR_1D,
            ),
            None,
            None,
        )


def broadcast_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _Broadcast1D.apply(inputs, parallel_context)


def all_reduce_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _AllReduce1D.apply(inputs, parallel_context)


def all_gather_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _AllGather1D.apply(inputs, dim, parallel_context)


def scatter_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _Scatter1D.apply(inputs, dim, parallel_context)
