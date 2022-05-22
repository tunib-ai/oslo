import copy

import torch
import torch.nn as nn
from torch.nn import Embedding

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding1D,
    VocabUtility,
)
from oslo.torch.nn.modules.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from oslo.torch.nn.parallel.tensor_parallel.mapping import (
    TensorParallelMapping,
)
from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    _update_module_arguments,
    is_huggingface_model,
    is_oslo_model,
)
from oslo.transformers.mapping_utils import (
    _TensorParallelMappingForHuggingFace,
)


class _TensorParallel1D(ParallelWrapper):
    """
    PyTorch module for 1D tensor parallelism

    Args:
        module (nn.Module): model object
        parallel_context (ParallelContext): parallel context object
        mapping (dict): custom tensor parallel mapping
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

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def _parallelize(self):
        self._update_mp_arguments()
        self._parallelize_embedding()
        self._parallelize_linear()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    def _update_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    world_size = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_1D
                    )
                    reduced_arg = getattr(module, elem.name) // world_size
                    setattr(module, elem.name, reduced_arg)

    @staticmethod
    def _deconstruct_combined_qkv(tensor, world_size, fusion_degree):
        tensor = [
            tensor[i * world_size : (i + 1) * world_size] for i in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
        return tensor

    def _slice(self, module, reversed, fusion_degree, slice_bias, dim):
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            reversed=reversed,
            fusion_degree=fusion_degree,
            orig_module=copy.deepcopy(module.__class__),
            gather_output=False,
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:

                if reversed:
                    module.weight.data = module.weight.data.t()

                weight_list = module.weight.data.chunk(
                    fusion_degree * world_size, dim=dim
                )

                if fusion_degree > 1:
                    weight_list = self._deconstruct_combined_qkv(
                        weight_list,
                        world_size,
                        fusion_degree,
                    )

                module.weight.data = weight_list[rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_1D] = rank
                else:
                    module.weight.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[1],
                out_features=module.weight.size()[0],
            )

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.chunk(fusion_degree * world_size, dim=0)

                if fusion_degree > 1:
                    bias_list = self._deconstruct_combined_qkv(
                        bias_list,
                        world_size,
                        fusion_degree,
                    )

                module.bias.data = bias_list[rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_1D] = rank
                else:
                    module.bias.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

        return module

    def _column_slice(self, module: nn.Module, fusion_degree: int, reversed: bool):
        return self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=True,
            dim=0,
        )

    def _row_slice(self, module: nn.Module, fusion_degree: int, reversed: bool):
        return self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=False,
            dim=1,
        )

    def _parallelize_embedding(self):
        module = self.module
        while isinstance(module, ParallelWrapper):
            module = module.module

        assert hasattr(module, "get_input_embeddings"), (
            "model object must have `get_input_embeddings` and "
            "`get_output_embeddings` method."
        )

        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        embedding = module.get_input_embeddings()

        (
            vocab_start_index,
            vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            embedding.num_embeddings, rank, world_size
        )

        weight_list = embedding.weight.chunk(world_size, dim=0)

        embedding.weight.data = weight_list[rank].contiguous()

        if hasattr(embedding.weight, "oslo_parallel"):
            embedding.weight.oslo_parallel[ParallelMode.TENSOR_1D] = rank
        else:
            embedding.weight.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

        _update_module_arguments(
            module=embedding,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
            parallel_context=self.parallel_context,
            num_embeddings=embedding.weight.size()[0],
            orig_module=copy.deepcopy(embedding.__class__),
        )

        if isinstance(embedding, Embedding):
            embedding.__class__ = VocabParallelEmbedding1D

        for name, _module in module.named_modules():
            if (
                hasattr(_module, "weight")
                and _module.weight is embedding.weight
                and not isinstance(_module, Embedding)
            ):
                _update_module_arguments(
                    module=_module,
                    parallel_context=self.parallel_context,
                    reversed=self.tensor_parallel_mapping.is_reversed_param(
                        self.module, name
                    ),
                    fusion_degree=1,
                    orig_module=copy.deepcopy(_module.__class__),
                    gather_output=not is_oslo_model(self.module),
                    out_features=embedding.weight.size()[0],
                )

                if isinstance(_module, nn.Linear):
                    _module.__class__ = ColumnParallelLinear
                else:
                    raise RuntimeError("Classifier layer must be `nn.Linear` class")

    def _parallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(self.module, param_name):
                self._column_slice(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed_param(
                        self.module, param_name
                    ),
                    fusion_degree=self.tensor_parallel_mapping.get_combined_qkv_degree(
                        self.module, param_name, module
                    ),
                )
                module.__class__ = ColumnParallelLinear

            elif self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                self._row_slice(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed_param(
                        self.module, param_name
                    ),
                    fusion_degree=1,
                )
                module.__class__ = RowParallelLinear
