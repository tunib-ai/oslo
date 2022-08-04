from typing import Any

import torch
from torch import Tensor

from oslo.torch.distributed import ParallelMode, ParallelContext
from oslo.torch.distributed.nn.functional import all_gather, all_reduce, scatter


class _BroadcastTensor1D(torch.autograd.Function):
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


class _AllReduceTensor1D(torch.autograd.Function):
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


class _AllGatherTensor1D(torch.autograd.Function):
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


class _ScatterTensor1D(torch.autograd.Function):
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


def broadcast_tensor_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _BroadcastTensor1D.apply(inputs, parallel_context)


def all_reduce_tensor_1d(inputs: Tensor, parallel_context: ParallelContext):
    return _AllReduceTensor1D.apply(inputs, parallel_context)


def all_gather_tensor_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _AllGatherTensor1D.apply(inputs, dim, parallel_context)


def scatter_tensor_1d(inputs: Tensor, dim: int, parallel_context: ParallelContext):
    return _ScatterTensor1D.apply(inputs, dim, parallel_context)
