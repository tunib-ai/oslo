import torch

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._ops import all_gather, all_reduce, scatter


class _Broadcast1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return inputs

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return (
            all_reduce(
                grad,
                parallel_mode=ParallelMode.TENSOR_1D,
                parallel_context=parallel_context,
                on_cpu=str(grad.device) == "cpu",
                async_op=False,
            ),
            None,
        )


class _AllReduce1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        return all_reduce(
            inputs,
            parallel_mode=ParallelMode.TENSOR_1D,
            parallel_context=parallel_context,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
        )

    def backward(ctx, grad):
        return grad, None


class _AllGather1D(torch.autograd.Function):
    def forward(ctx, inputs, parallel_context):
        ctx.parallel_context = parallel_context
        return all_gather(
            inputs,
            dim=-1,
            parallel_mode=ParallelMode.TENSOR_1D,
            parallel_context=parallel_context,
            on_cpu=str(inputs.device) == "cpu",
            async_op=False,
        )

    def backward(ctx, grad):
        parallel_context = ctx.parallel_context
        return scatter(grad, parallel_context), None


def broadcast_1d(inputs, parallel_context):
    return _Broadcast1D.apply(inputs, parallel_context)


def all_reduce_1d(inputs, parallel_context):
    return _AllReduce1D.apply(inputs, parallel_context)


def all_gather_1d(inputs, parallel_context):
    return _AllGather1D.apply(inputs, parallel_context)
