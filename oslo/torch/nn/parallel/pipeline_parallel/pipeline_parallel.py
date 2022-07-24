import time     # TODO; temp
import concurrent.futures
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import rpc

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed.nn.functional import broadcast
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.utils import get_parallel_context

from ._server import workers, _ORIGINAL_FORWARDS, _NEW_FORWARDS
from ._communicator import FINAL_RESULT_QUEUE, REMOTE_JOB_QUEUE, push_result_queue, push_job_queue, push_final_result_queue


class PipelineParallel(nn.Module):
    """
    Pipeline parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
        memory_computation_balance (float): memory computation balance factor

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`.
        2. Support multiple scheduling algorithms.
        3. Support inter-module partitioning described in Sagemaker Model Parallelism.

    Examples:
        >>> from oslo.torch.nn.parallel import PipelineParallel
        >>>
        >>> model = AnyPytorchModel()
        >>> optimizer = AnyOptimizer(model.parameters(), lr=3e-5)
        >>> pp_wrapper = PipelineParallel(model, ...)

        >>> output = pp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: ParallelContext,
        memory_computation_balance: float = 1.0,
        tracing_inputs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.partitioner = ModelPartitioner(
            module=module,
            process_group=parallel_context.get_group(ParallelMode.PIPELINE),
            tracing_inputs=tracing_inputs,
            memory_computation_balance=memory_computation_balance,
        )
        self.partitioner.partition()
        self.oslo_parallel = self.module.oslo_parallel      # TODO; right?

        self._forward_dict = dict()
        self.build_forward_dict()

    def forward(self, x):
        # TODO;
        #   need to make event that counts root node's
        #   number of forward input-output diff and backward input-output diff.
        #   otherwise, we cannot know that forward-backward is done.

        rank = dist.get_rank()
        num_mb = 2  # TODO;

        # TODO;
        if rank == 0:
            micro_batches = x.chunk(num_mb)
            futures = []
            for mb in micro_batches:
                future = workers.put(self.module, mb)
                futures.append(future)

            results = []
            for done in concurrent.futures.as_completed(futures):
                results.append(done.result())

            for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                if other == 0:
                    continue

                end_call = rpc.rpc_async(
                    to=f'PP_WORKER_{other}',
                    func=push_job_queue,
                    args=(None, ),
                )
                end_call.wait()

            results = torch.cat(results, 0)
            for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                if other == 0:
                    continue

                end_call = rpc.rpc_async(
                    to=f'PP_WORKER_{other}',
                    func=push_final_result_queue,
                    args=(results, ),
                )
                end_call.wait()

        else:
            while True:     # TODO; better way?
                job = REMOTE_JOB_QUEUE.get()

                if job is None:
                    break

                inputs, msg = job
                src = msg.src
                location = msg.location
                tag = msg.id

                forward_fn = _ORIGINAL_FORWARDS[location]
                result = forward_fn(inputs)

                rpc_dst = f'PP_WORKER_{src.index}'  # TODO;

                rpc.rpc_async(
                    to=rpc_dst,
                    func=push_result_queue,
                    args=(result, tag),
                )

            # forward pass end, get copy from master
            results = FINAL_RESULT_QUEUE.get()

        return results

    def build_forward_dict(self):

        def get_location(module, prefix):
            self._forward_dict[prefix] = module
            setattr(module, "location", prefix)
            setattr(module, "parallel_context", self.parallel_context)

            for n, m in module.named_children():
                new_prefix = f'{prefix}.{n}' if prefix != '' else n
                get_location(m, new_prefix)

        get_location(self, "")
