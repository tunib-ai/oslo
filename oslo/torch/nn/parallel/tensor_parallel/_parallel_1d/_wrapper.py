import copy

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding1D,
    Embedding1D,
    VocabUtility,
)
from oslo.torch.nn.modules.linear import (
    ColLinear1D,
    RowLinear1D,
)
from oslo.torch.nn.modules.layer_norm import (
    LayerNorm1D,
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
        self._parallelize_layernorm()
        self._parallelize_head()
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

    def _parallelize_embedding(self):
        for module in self.module.modules():
            if isinstance(module, nn.Embedding):
                self._slice_embedding(
                    module=module,
                )

    def _parallelize_layernorm(self):
        for module in self.module.modules():
            if isinstance(module, nn.LayerNorm):
                self._slice_layernorm(
                    module=module,
                )

    def _parallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(self.module, param_name):
                self._column_slice_linear(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed(
                        self.module, param_name
                    ),
                    fusion_degree=self.tensor_parallel_mapping.get_combined_qkv_degree(
                        self.module, param_name, module
                    ),
                    gather_output=self.tensor_parallel_mapping.is_gather_output(
                        self.module, param_name
                    ),
                )

            elif self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                self._row_slice_linear(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed(
                        self.module, param_name
                    ),
                    fusion_degree=1,
                )

    def _parallelize_head(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_head(
                self.module, param_name
            ) and isinstance(module, nn.Linear):
                self._slice_head(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed(
                        self.module, param_name
                    ),
                )

    @staticmethod
    def _deconstruct_combined_qkv(tensor, world_size, fusion_degree, dim):
        tensor = [
            tensor[i * world_size : (i + 1) * world_size] for i in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=dim), zip(*tensor)))
        return tensor

    def _slice_embedding(self, module):
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        if module is self.module.get_input_embeddings():

            (
                vocab_start_index,
                vocab_end_index,
            ) = VocabUtility.vocab_range_from_global_vocab_size(
                module.num_embeddings, rank, world_size
            )

            weight_list = module.weight.chunk(world_size, dim=0)

            module.weight.data = weight_list[rank].contiguous()

            _update_module_arguments(
                module=module,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                parallel_context=self.parallel_context,
                num_embeddings=module.weight.size()[0],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = VocabParallelEmbedding1D
        else:
            weight_list = module.weight.data.chunk(world_size, dim=1)
            module.weight.data = weight_list[rank].contiguous()

            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = Embedding1D

        if hasattr(module.weight, "oslo_parallel"):
            module.weight.oslo_parallel[ParallelMode.TENSOR_1D] = rank
        else:
            module.weight.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

    def _slice_linear(
        self,
        module: nn.Module,
        reversed: bool,
        fusion_degree: int,
        slice_bias: bool,
        dim: int,
    ):
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)

        if reversed:
            module.weight.data = module.weight.data.t()

        weight_list = module.weight.data.chunk(fusion_degree * world_size, dim=dim)

        if fusion_degree > 1:
            weight_list = self._deconstruct_combined_qkv(
                weight_list,
                world_size,
                fusion_degree,
                dim=dim,
            )

        module.weight.data = weight_list[rank].contiguous()

        if hasattr(module.weight, "oslo_parallel"):
            module.weight.oslo_parallel[ParallelMode.TENSOR_1D] = rank
        else:
            module.weight.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.chunk(fusion_degree * world_size, dim=0)

                if fusion_degree > 1:
                    bias_list = self._deconstruct_combined_qkv(
                        bias_list,
                        world_size,
                        fusion_degree,
                        dim=0,
                    )

                module.bias.data = bias_list[rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_1D] = rank
                else:
                    module.bias.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

    def _column_slice_linear(
        self,
        module: nn.Module,
        reversed: bool,
        fusion_degree: int,
        gather_output: bool,
    ):
        self._slice_linear(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=True,
            dim=0,
        )

        _update_module_arguments(
            module=module,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_context=self.parallel_context,
            reversed=reversed,
            fusion_degree=fusion_degree,
            orig_module=copy.deepcopy(module.__class__),
            gather_output=gather_output,
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
        )
        module.__class__ = ColLinear1D

    def _row_slice_linear(self, module: nn.Module, reversed: bool, fusion_degree: int):
        self._slice_linear(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=False,
            dim=1,
        )
        _update_module_arguments(
            module=module,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_context=self.parallel_context,
            reversed=reversed,
            fusion_degree=fusion_degree,
            orig_module=copy.deepcopy(module.__class__),
            parallel_input=True,
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
        )
        module.__class__ = RowLinear1D

    def _slice_layernorm(self, module):
        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0],
            partitioned_dim=module.weight.size()[0],
            parallel_context=self.parallel_context,
            orig_module=copy.deepcopy(module.__class__),
        )
        module.__class__ = LayerNorm1D

    def _slice_head(self, module, reversed):
        if module.weight is not self.module.get_input_embeddings().weight:
            self._column_slice_linear(
                module=module,
                reversed=reversed,
                fusion_degree=1,
                gather_output=not is_oslo_model(self.module),
            )
        else:
            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                reversed=reversed,
                fusion_degree=1,
                orig_module=copy.deepcopy(module.__class__),
                gather_output=not is_oslo_model(self.module),
                skip_bias_add=module.skip_bias_add
                if hasattr(module, "skip_bias_add")
                else False,
            )
        module.__class__ = ColLinear1D
