from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._model_partitioner import ModelPartitioner
from oslo.torch.nn.parallel.utils import get_parallel_context

#from oslo.torch.nn.parallel.pipeline_parallel._wrapper import (
#    _PipelineParallel,
#)


class PipelineParallel(nn.Module):
    """
    Pipeline parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process group object
        memory_computation_balance (float): memory computation balance factor
        micro_batch_size (int): micro batch size

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
        micro_batch_size: int,
        use_auto_partitioning: bool = True,
        memory_computation_balance: float = 1.0,
        tracing_inputs: Dict[str, Any] = None,
    ):
        super().__init__()
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context)
        self.partitioner = ModelPartitioner(
            module=module,
            parallel_context=self.parallel_context,
            tracing_inputs=tracing_inputs,
            memory_computation_balance=memory_computation_balance,
        )
        self.device = torch.cuda.current_device()
        self.batch_size = None
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = None
        self.micro_batches = None

        if use_auto_partitioning:
            self.partitioner.partition()

        #self.module = _pipeline_wrapper()

    def forward(self, *args, **kwargs):

        #TODO: Do we need a Zero stage check here?



        #self.micro_batches = self._split_batches(kwargs)

        

        return self.module(*args, **kwargs)


    def guess_batch_size(kwargs):
        """Guess global batch size dynamically from user input"""
        for key in ["input_ids", "attention_mask", "labels"]:
            if kwargs.get(key, None) is not None:
                assert torch.is_tensor(
                    kwargs.get(key)
                ), f"Param ``{key}`` must be ``torch.Tensor`` that has shape like (batch_size, ...)."

                return kwargs.get(key).size(0)

        kwargs_types = {k: type(v).__qualname__ for k, v in kwargs.items()}

        raise ValueError(
            f"You must at least input one of ``input_ids``, ``attention_mask`` or ``labels``. "
            f"But you didn't input any of them. Please double check your input: {kwargs_types}."
        )

    def _split_batches(self, batches):
        """Split mini-batches to micro-batches"""
        self.batch_size = self.guess_batch_size(batches)
        assert self.batch_size % self.micro_batch_size == 0, (
            "``micro_batch_size`` must be divisible by batch size. "
            f"currently, ``micro_batch_size`` is {self.micro_batch_size}. "
            f"but batch size is {self.batch_size}."
        )

        self.num_micro_batches = self.batch_size // self.micro_batch_size
        _micro_batches = [{} for _ in range(self.num_micro_batches)]

        for k, v in batches.items():
            if torch.is_tensor(v):
                if v.size(0) == self.batch_size:
                    micro_batch = v.chunk(self.num_micro_batches, dim=0)
                    for i, m in enumerate(micro_batch):
                        _micro_batches[i][k] = m
                else:
                    for i in range(self.num_micro_batches):
                        _micro_batches[i][k] = v
            else:
                for i in range(self.num_micro_batches):
                    _micro_batches[i][k] = v

        return _micro_batches