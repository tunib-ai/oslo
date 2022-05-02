from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.utils import ParallelWrapper, get_parallel_context


class _SequenceDataParallelState(object):
    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context


def _sequence_data_parallel_hook(
    state: _SequenceDataParallelState, bucket: dist._GradBucket
) -> torch.futures.Future:
    parallel_context = state.parallel_context
    group_to_use = parallel_context.get_group(ParallelMode.SEQUENCE_DP)
    div_factor = parallel_context.get_world_size(ParallelMode.DATA)

    # divide the tensor with DP size
    tensor = bucket.get_tensors()[0]
    tensor.div_(div_factor)

    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()
    return fut


class SequenceDataParallel(DistributedDataParallel, ParallelWrapper):
    def __init__(
        self,
        module,
        parallel_context: Optional[ParallelContext] = None,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
    ):
        self.parallel_context = get_parallel_context(module, parallel_context)
        super(SequenceDataParallel, self).__init__(
            module,
            process_group=self.parallel_context.get_group(ParallelMode.SEQUENCE_DP),
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        self.register_comm_hook(
            state=_SequenceDataParallelState(self.parallel_context),
            hook=_sequence_data_parallel_hook,
        )
