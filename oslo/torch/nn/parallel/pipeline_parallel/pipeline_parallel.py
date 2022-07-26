import concurrent.futures
from queue import Queue
from threading import Lock
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import rpc

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.utils import get_parallel_context

from ._functional import pipe_backward_redirection
from ._messages import generate_request
from ._server import (
    workers,
    _MODULE_DEVICE_LOCATIONS, _ORIGINAL_FORWARDS, ACTIVATIONS,
    FINAL_RESULT_QUEUE, REMOTE_JOB_QUEUE, REMOTE_RESULT_QUEUES,
    push_result_queue, push_job_queue, push_final_result_queue,
)


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
        self.oslo_parallel = self.module.oslo_parallel
        self._recursive_wrap(self, "")
        self._lock = Lock()

    def forward(self, x):
        rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
        num_mb = 1  # TODO;

        if rank == 0:
            micro_batches = x.chunk(num_mb)     # TODO;
            futures = []
            for mb in micro_batches:
                future = workers.put(self.module, mb)
                futures.append(future)

            results = []
            for done in concurrent.futures.as_completed(futures):
                results.append(done.result())

            # TODO; why putting None and result at the same time make a strange result?
            for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                if other == rank:
                    continue

                end_call = rpc.rpc_async(
                    to=f'PP_WORKER_{other}',
                    func=push_job_queue,
                    args=(None, ),
                )
                end_call.wait()

            results = torch.cat(results, 0)
            for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                if other == rank:
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
                req, args, kwargs = job

                if req is None:
                    break

                args = pipe_backward_redirection(req, *args)

                src = req.src
                dst = req.dst
                location = req.location
                tag = req.tag

                forward_fn = _ORIGINAL_FORWARDS[location]
                result = forward_fn(*args, **kwargs)

                # make reverse direction req for backward
                rpc_caller = self.parallel_context.get_pipeline_rpc_worker_name(dst.index)
                with self._lock:  # TODO; need a lock?
                    reverse_req = generate_request(dst, src, None, rpc_caller)

                rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(src.index)
                rpc.rpc_async(
                    to=rpc_dst,
                    func=push_result_queue,
                    args=(reverse_req, result, tag),
                )

                reverse_tag = reverse_req.tag
                assert reverse_tag not in ACTIVATIONS      # TODO; check tag duplicate in other way
                # ACTIVATIONS[reverse_tag] = Queue()
                # ACTIVATIONS[reverse_tag].put(result)

                ACTIVATIONS[reverse_tag] = result

            # forward pass end, get copy from master
            results = FINAL_RESULT_QUEUE.get()

        return results

    def _recursive_wrap(self, module, prefix):
        setattr(module, "location", prefix)
        if prefix != "":    # wrapper's forward function should not be wrapped
            self._wrap_forward(module)

        for name, m in module.named_children():
            new_prefix = f'{prefix}.{name}' if prefix != '' else name
            self._recursive_wrap(m, new_prefix)

    def _wrap_forward(self, module):
        orig_forward = module.forward
        loc = module.location
        device = module.oslo_parallel[ParallelMode.PIPELINE]

        _ORIGINAL_FORWARDS[loc] = orig_forward
        _MODULE_DEVICE_LOCATIONS[loc] = device

        def new_forward(*args, **kwargs):
            location = module.location

            module_device = _MODULE_DEVICE_LOCATIONS[location]
            module_device = torch.device('cuda', module_device)
            current_device = torch.cuda.current_device()
            current_device = torch.device('cuda', current_device)
            is_same = module_device == current_device

            if is_same:
                # just put task in the job Queue
                forward_fn = _ORIGINAL_FORWARDS[location]

                future = workers.put(forward_fn, *args, **kwargs)
                result = future.result()

            else:
                src = dist.get_rank()
                src = torch.device('cuda', src)
                dst = _MODULE_DEVICE_LOCATIONS[location]
                dst = torch.device('cuda', dst)
                rpc_caller = self.parallel_context.get_pipeline_rpc_worker_name(src.index)

                with self._lock:    # TODO; need a lock?
                    req = generate_request(src, dst, location, rpc_caller)
                tag = req.tag
                REMOTE_RESULT_QUEUES[tag] = Queue()
                # ACTIVATIONS[tag] = Queue()
                # ACTIVATIONS[tag].put((args, kwargs))
                ACTIVATIONS[tag] = args     # TODO; kwargs...

                rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(dst.index)
                rpc.rpc_async(
                    to=rpc_dst,
                    func=push_job_queue,
                    args=(req, ) + args,
                    kwargs=kwargs,
                )

                result = REMOTE_RESULT_QUEUES[tag].get()

               # pre-work for backward
                req, result = result
                # TODO; avoid wrap by tuple
                wrapped = False
                if isinstance(result, torch.Tensor):
                    result = (result, )
                    wrapped = True
                result = pipe_backward_redirection(req, *result)
                if wrapped:
                    result = result[0]

                del REMOTE_RESULT_QUEUES[tag]

            return result

        module.forward = new_forward
