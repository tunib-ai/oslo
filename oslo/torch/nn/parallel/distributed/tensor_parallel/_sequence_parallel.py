import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from oslo.torch.distributed import ParallelContext, ParallelMode


class SequenceParallelState(object):
    def __init__(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context


def _sequence_parallel_hook(state: SequenceParallelState, bucket: dist._GradBucket) -> torch.futures.Future:
    parallel_context = state.parallel_context
    group_to_use = parallel_context.get_group(ParallelMode.SEQUENCE_DP)
    div_factor = parallel_context.get_world_size(ParallelMode.DATA)

    # divide the tensor with DP size
    tensor = bucket.get_tensors()[0]
    tensor.div_(div_factor)

    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()
    return fut


class DistributedSequenceParallel(DistributedDataParallel):
    def __init__(
        self,
        module, parallel_context: ParallelContext,
        device_ids=None,
        output_device=None,
        dim=0, broadcast_buffers=True,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False
    ):
        self._parallel_context = parallel_context
        super(DistributedSequenceParallel, self).__init__(
            module,
            process_group=self._parallel_context.get_group(ParallelMode.SEQUENCE_DP),
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
            state=SequenceParallelState(self._parallel_context),
            hook=_sequence_parallel_hook,
        )
