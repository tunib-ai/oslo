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


def get_divisible_by(parallel_context: ParallelContext):
    if parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_1D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL) ** 2
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2P5D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
    elif parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_3D:
        divisible_by = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT) ** 2
    else:
        raise ValueError(
            "currently, only 1D, 2D, 2.5D tensor parallelism is supported."
        )
    return divisible_by


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

        divisible_by = get_divisible_by(parallel_context)
        while new_vocab_size % divisible_by != 0:
            new_vocab_size += 1

        if new_vocab_size != vocab_size:
            padding = torch.zeros(
                new_vocab_size - vocab_size,
                embedding_dim,
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
            new_embeddings = torch.cat(
                tensors=[module.weight.data, padding],
                dim=0,
            )

            module.weight.data = new_embeddings
            module.num_embeddings = new_vocab_size
            setattr(unwrapped_model, "orig_vocab_size", vocab_size)
        return model

    @staticmethod
    def _resize_num_classes(model, parallel_context, mapping):
        unwrapped_model = unwrap_parallel(model)
        if mapping is None:
            if is_huggingface_model(unwrapped_model):
                mapping = _TensorParallelMappingForHuggingFace().get_mapping(
                    unwrapped_model
                )
            else:
                raise ValueError(
                    "`mapping` must be input if the model is not huggingface model."
                )
        tensor_parallel_mapping = TensorParallelMapping(mapping)
        divisible_by = get_divisible_by(parallel_context)

        for param_name, module in unwrapped_model.named_modules():
            if tensor_parallel_mapping.is_head(
                unwrapped_model, param_name
            ) and isinstance(module, nn.Linear):
                if module.weight is unwrapped_model.get_input_embeddings().weight:
                    module.out_features = (
                        unwrapped_model.get_input_embeddings().num_embeddings
                    )
                else:
                    out_features, in_features = module.weight.size()
                    new_out_features = out_features

                    while new_out_features % divisible_by != 0:
                        new_out_features += 1

                    if new_out_features != out_features:
                        padding = torch.zeros(
                            new_out_features - out_features,
                            in_features,
                            dtype=module.weight.dtype,
                            device=module.weight.device,
                        )
                        new_weight = torch.cat(
                            tensors=[module.weight.data, padding],
                            dim=0,
                        )

                        if hasattr(module, "bias") and module.bias is not None:
                            padding = torch.zeros(
                                new_out_features - out_features,
                                dtype=module.weight.dtype,
                                device=module.weight.device,
                            )
                            new_bias = torch.cat(
                                tensors=[module.bias.data, padding],
                                dim=0,
                            )
                            module.bias.data = new_bias

                        module.weight.data = new_weight
                        module.out_features = new_out_features

                        setattr(
                            unwrapped_model,
                            f"orig_{param_name.split('.')[-1]}_num_classes",
                            out_features,
                        )
                        if hasattr(unwrapped_model, "num_labels"):
                            unwrapped_model.num_labels = new_out_features
        return model

    def _remove_embeddings(self, model, parallel_context):
        pass
