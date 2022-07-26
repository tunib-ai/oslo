import concurrent.futures
import threading
import time
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

from ._functional import request_backward_redirection, response_backward_redirection
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
        num_micro_batches:int = 1,
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
        self.rank = self.parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.num_micro_batches = num_micro_batches

    def forward(self, x):
        futures = []
        if self.rank == 0:
            micro_batches = x.chunk(self.num_micro_batches)     # TODO;

            def notify(fut):
                for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                    rpc.rpc_async(
                        to=f'PP_WORKER_{other}',
                        func=push_job_queue,
                        args=(None, ),
                    )

            for ind, mb in enumerate(micro_batches):
                future = workers.put(self.module, mb)
                if ind+1 == self.num_micro_batches:
                    future.add_done_callback(notify)
                futures.append(future)

        # TODO; 주석 영어로 번역..
        #  inner_loop 함수는 아무 것도 return 하지 않지만
        #  new_forward 함수에서 workers에 return 해준다.
        self._inner_loop()

        if self.rank == 0:
            for done in concurrent.futures.as_completed(futures):
                result = done.result()

                # distribute
                for other in self.parallel_context.get_ranks_in_group(ParallelMode.PIPELINE):
                    if other == self.rank:
                        continue

                    rpc_dst = self.parallel_context.get_pipeline_rpc_worker_name(other)
                    rpc.rpc_async(
                        to=rpc_dst,
                        func=push_final_result_queue,
                        args=(result, ),
                    ).wait()
                yield result

        else:
            # forward pass end, wait results from master
            for _ in range(self.num_micro_batches):
                result = FINAL_RESULT_QUEUE.get()
                yield result    # has no gradient

    def _inner_loop(self):
        while True:     # TODO; better way?
            job = REMOTE_JOB_QUEUE.get()
            req, args, kwargs = job

            if req is None:
                break

            args = request_backward_redirection(req, *args)

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

            ACTIVATIONS[reverse_tag] = result

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

            print(location)

            module_device = _MODULE_DEVICE_LOCATIONS[location]
            module_device = torch.device('cuda', module_device)
            current_device = torch.cuda.current_device()
            current_device = torch.device('cuda', current_device)
            is_same = module_device == current_device

            if is_same:
                # just put task in the job Queue
                forward_fn = _ORIGINAL_FORWARDS[location]
                result = forward_fn(*args, **kwargs)

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

                print(f'{result=}')

                # pre-work for backward
                req, result = result
                # TODO; avoid wrap by tuple
                wrapped = False
                if isinstance(result, torch.Tensor):
                    result = (result, )
                    wrapped = True
                result = response_backward_redirection(req, *result)
                if wrapped:
                    result = result[0]

                del REMOTE_RESULT_QUEUES[tag]

            return result

        module.forward = new_forward
