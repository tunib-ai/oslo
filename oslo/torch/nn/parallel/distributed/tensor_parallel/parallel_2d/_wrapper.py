import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel.distributed.tensor_parallel.parallel_1d._mapping import (
    TensorParallelMapping,
)
from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    is_huggingface_model,
)
from oslo.transformers.mapping_utils import _TensorParallelMappingForHuggingFace


class _TensorParallel2D(ParallelWrapper):
    """
    PyTorch module for 2D tensor parallelism

    Args:
        module (nn.Module): model object
        parallel_context (ParallelContext): parallel context object
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: ParallelContext,
        mapping: dict = None,
    ):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context
        self.device = torch.cuda.current_device()

        if mapping is None:
            if is_huggingface_model(module):
                mapping = _TensorParallelMappingForHuggingFace().get_mapping(module)
            else:
                raise ValueError(
                    "`mapping` must be input if the model is not huggingface model."
                )

        self.tensor_parallel_mapping = TensorParallelMapping(mapping)
        self._parallelize()

    def _parallelize(self):
        pass

    def _slice(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(
                self.module, param_name
            ) or self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                pass

    def _update_mp_arguments(self):
        pass

    def _parallelize_embedding(self):
        pass

    def _parallelize_modules(self):
        pass
