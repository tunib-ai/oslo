from colossalai.core import global_context as gpc
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine.gradient_handler._base_gradient_handler import BaseGradientHandler
from colossalai.context.parallel_mode import ParallelMode
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import Iterable


def bucket_allreduce(param_list: Iterable[nn.Parameter], group=None, average=False):
    # get communication world size
    comm_size = dist.get_world_size(group)
    # bucketize and all-reduce
    buckets = {}
    # Pack the buckets.
    for param in param_list:
        if param.requires_grad and param.grad is not None:
            tp = param.data.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(param)

    # For each bucket, all-reduce and copy all-reduced grads.
    for tp in buckets:
        bucket = buckets[tp]
        grads = [param.grad.data for param in bucket]
        coalesced = _flatten_dense_tensors(grads)
        if average:
            coalesced /= comm_size

        dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=group)
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


@GRADIENT_HANDLER.register_module
class SequenceParallelGradientHandlerTest(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group.
    A all-reduce collective communication will be operated in
    :func:`handle_gradient` among a data parallel group.
    For better performance, it bucketizes the gradients of all parameters that are
    the same type to improve the efficiency of communication.
    """

    def handle_gradient(self):
        """A method running a all-reduce operation in a data parallel group.
        """
        if gpc.get_world_size(ParallelMode.SEQUENCE) > 1:
            print('SP reduce gradient!')
            bucket_allreduce(param_list=self._model.parameters(), group=gpc.get_group(ParallelMode.SEQUENCE))

        dist.barrier()

        if gpc.get_world_size(ParallelMode.DATA) > 1:
            print('DP reduce gradient!')
            bucket_allreduce(param_list=self._model.parameters(), group=gpc.get_group(ParallelMode.DATA), average=True)
