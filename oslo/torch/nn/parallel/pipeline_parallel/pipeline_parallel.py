import time     # TODO; temp
from typing import Optional, Dict, Any

import torch.nn as nn
import torch.distributed as dist

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.utils import get_parallel_context


from ._hooks import rpc_push_queue, rpc_pull_queue


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
        self.build_forward_dict()

    def forward(self, x):
        num_micro_batches = 2
        # micro_batches = x.chunk(num_micro_batches)

        # TODO;
        #   need to make event that counts root node's
        #   number of forward input-output diff and backward input-output diff.
        #   otherwise, we cannot know that forward-backward is done.

        rank = dist.get_rank()

        if rank == 0:   # TODO; and not tail?
            self.event_head(x)
        # TODO; IMPORTANT! make tail and prepare backprop, wait until work ends
        else:
            self.inner_loop()

    def event_head(self, x):
        self.module(x)      # TODO; need layer-wise split or check
        self.inner_loop()

    def inner_loop(self):
        print("INNER LOOP")
        msg = rpc_pull_queue()
        print(msg)
        # TODO; async recv of real input tensor
        forward_fn = self._forward_dict[msg.location]

    def build_forward_dict(self):
        self._forward_dict = dict()

        def get_location(module, prefix):
            self._forward_dict[prefix] = module.forward
            setattr(module, "_location", prefix)

            for n, m in module.named_children():
                new_prefix = f'{prefix}.{n}' if prefix != '' else n
                get_location(m, new_prefix)

        get_location(self.module, "")
