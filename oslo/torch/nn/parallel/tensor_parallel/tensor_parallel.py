from typing import Optional

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._wrapper import (
    _TensorParallel1D,
)
from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._wrapper import (
    _TensorParallel2D,
)
from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._wrapper import (
    _TensorParallel2p5D,
)
from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    unwrap_parallel,
    get_parallel_context,
)


class TensorParallel(ParallelWrapper):
    """
    Tensor parallel module

    Args:
        module (nn.Module): PyTorch module object
        parallel_context (ParallelContext): process context

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`.
        2. Support auto de-parallelism

    Examples:
        >>> from oslo.torch.nn.parallel import TensorParallel

        >>> model = AnyTransformerModel()
        >>> optimizer = AnyOptimizer(model.parameters(), lr=3e-5)
        >>> tp_wrapper = TensorParallel(model, parallel_context=..., ...)

        >>> output = tp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: Optional[ParallelContext] = None,
    ):
        super().__init__()
        self.parallel_context = get_parallel_context(module, parallel_context)
        orig_vocab_size, module = self._add_embeddings(module, self.parallel_context)
        if self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_1D:
            self.module = _TensorParallel1D(module, self.parallel_context)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
            self.module = _TensorParallel2D(module, self.parallel_context)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2P5D:
            self.module = _TensorParallel2p5D(module, self.parallel_context)
        else:
            raise ValueError("currently, only 1d tensor parallelism is supported.")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @staticmethod
    def _add_embeddings(model, parallel_context):
        unwrap = unwrap_parallel(model)

        assert hasattr(
            unwrap, "get_input_embeddings"
        ), "model object must have `get_input_embeddings` method."

        input_embeddings = unwrap.get_input_embeddings()

        emb_size, dimension = input_embeddings.weight.size()
        new_emb_size = emb_size
        world_size = parallel_context.get_world_size(ParallelMode.TENSOR)

        while new_emb_size % world_size != 0:
            new_emb_size += 1

        padding = torch.zeros(
            new_emb_size - emb_size,
            dimension,
            dtype=input_embeddings.weight.dtype,
            device=input_embeddings.weight.device,
        )
        new_embeddings = torch.cat(
            tensors=[input_embeddings.weight, padding],
            dim=0,
        )

        new_num_embeddings = new_embeddings.size(0)

        for name, _module in unwrap.named_modules():
            # process tied weights
            if (
                hasattr(_module, "weight")
                and _module.weight is input_embeddings.weight
                and not isinstance(_module, nn.Embedding)
            ):
                _module.weight.data = new_embeddings
                if hasattr(_module, "out_features"):
                    _module.out_features = new_num_embeddings
                elif hasattr(_module, "nf"):
                    _module.nf = new_num_embeddings

        input_embeddings.weight.data = new_embeddings
        input_embeddings.num_embeddings = new_num_embeddings

        return emb_size, model

    def _remove_embeddings(self, model, parallel_context):
        pass
