import time

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.cuda.amp import custom_fwd, custom_bwd

from ._server import run_request_backward, run_response_backward


class _Dummy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        return args

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


# TODO; why
#  forward(ctx, req, *args, **kwargs)
#  ...
#  return args, kwargs
#  does not work???
# based on https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/pipe/rpc.py#L53
class _PipeRequestRedirection(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, req, *args):
        ctx.req = req
        ctx.num_nones = 1   # counting req
        ctx.num_nones += len(args)
        # ctx.num_nones += len(kwargs)

        return args

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):

        print(f'backward!, {dist.get_rank()=}, {ctx.req=}')

        rpc.rpc_async(
            to=ctx.req.caller,
            func=run_request_backward,
            args=(ctx.req.tag, ctx.req.caller, *grad_outputs),
        )

        return (None, ) * ctx.num_nones


def request_backward_redirection(req, *args, **kwargs):
    print(f'request_backward_redirection, {req=}, {args=}')
    return _PipeRequestRedirection.apply(req, *args, **kwargs)


class _PipeResponseRedirection(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, req, *args):
        ctx.req = req
        ctx.num_nones = 1   # counting req
        ctx.num_nones += len(args)
        # ctx.num_nones += len(kwargs)

        return args

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_outputs):

        print(f'backward!, {dist.get_rank()=}, {ctx.req=}')

        rpc.rpc_async(
            to=ctx.req.caller,
            func=run_response_backward,
            args=(ctx.req.tag, ctx.req.caller, *grad_outputs),
        )

        return (None, ) * ctx.num_nones


def response_backward_redirection(req, *args, **kwargs):
    print(f'response_backward_redirection, {req=}, {args=}')
    return _PipeResponseRedirection.apply(req, *args, **kwargs)
