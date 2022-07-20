import copy

import torch
import torch.nn as nn
import torch.distributed as dist

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
    _update_module_arguments,
    is_huggingface_model,
    is_oslo_model,
)
from oslo.torch.nn.parallel.tensor_parallel._base_wrapper import (
    BaseTensorParallelWrapper,
)
from oslo.transformers.mapping_utils import (
    _TensorParallelMappingForHuggingFace,
)


class _TensorParallel1D(BaseTensorParallelWrapper):
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
        module_args: dict = None,
    ):
        super().__init__(module, parallel_context)
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

        if module_args is None:
            if is_huggingface_model(module):
                module_args = module.config
            else:
                raise ValueError(
                    "`config` must be input if the model is not huggingface model."
                )

        self.config = module_args
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
                    assert (
                        getattr(module, elem.name) % world_size == 0
                    ), f"{elem.name} ({getattr(module, elem.name)}) must be divisible by world_size ({world_size})."
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
                world_size=world_size,
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
                world_size=world_size,
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
                bias_list = module.bias.data.chunk(fusion_degree * world_size, dim=0)

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
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
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
            world_size=world_size,
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
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
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
            world_size=world_size,
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
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0],
            partitioned_dim=module.weight.size()[0],
            parallel_context=self.parallel_context,
            world_size=world_size,
            orig_module=copy.deepcopy(module.__class__),
        )
        module.__class__ = LayerNorm1D

    def _slice_head(self, module, reversed):
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        if module.weight is not self.module.get_input_embeddings().weight:
            self._column_slice_linear(
                module=module,
                reversed=reversed,
                fusion_degree=1,
                gather_output=not is_oslo_model(self.module),
            )
        else:
            world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
            rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)

            if hasattr(module, "bias") and module.bias is not None:
                if module.bias.dim() >= 1:
                    bias_list = module.bias.data.chunk(world_size, dim=0)
                    module.bias.data = bias_list[rank].contiguous()

                    if hasattr(module.bias, "oslo_parallel"):
                        module.bias.oslo_parallel[ParallelMode.TENSOR_1D] = rank
                    else:
                        module.bias.oslo_parallel = {ParallelMode.TENSOR_1D: rank}

            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                world_size=world_size,
                reversed=reversed,
                fusion_degree=1,
                orig_module=copy.deepcopy(module.__class__),
                gather_output=not is_oslo_model(self.module),
                skip_bias_add=module.skip_bias_add
                if hasattr(module, "skip_bias_add")
                else False,
            )
        module.__class__ = ColLinear1D

    @torch.no_grad()
    def deparallelize(self):
        self._deparallelize_linear()
        self._deparallelize_embedding()
        self._rollback_mp_arguments()

    def _rollback_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    world_size = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_1D
                    )
                    expanded_arg = getattr(module, elem.name) * world_size
                    setattr(module, elem.name, expanded_arg)

    def _deparallelize_embedding(self):
        for param_name, module in self.module.named_modules():
            if module.__class__ == VocabParallelEmbedding1D:
                self._gather_embedding(module)
            if module.__class__ == Embedding1D:
                self._gather_embedding(module)

    def _deparallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(self.module, param_name):
                self._gather_column_linear(module)

            elif self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                self._gather_row_linear(module)

    def _gather_embedding(self, module):
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        if hasattr(module, "vocab_start_index") and hasattr(module, "vocab_end_index"):
            # w = gather_2d(self.parallel_context, module.weight.data, world_size, col_first=True)
            tensor_list = [
                torch.zeros_like(module.weight.data) for _ in range(world_size)
            ]
            dist.all_gather(
                tensor_list,
                module.weight.data.contiguous(),
                self.parallel_context.get_group(ParallelMode.TENSOR_1D),
            )
            w = torch.cat(tensor_list, dim=0)

            assert hasattr(
                self.module, "orig_vocab_size"
            ), "wrapper's vocab embedding module must have attribute 'orig_vocab_size'."
            orig_vocab_size = self.module.orig_vocab_size

            module.weight.data = w[:orig_vocab_size, :]

            _update_module_arguments(
                module=module,
                vocab_start_index=None,
                vocab_end_index=None,
                parallel_context=None,
                num_embeddings=module.weight.size()[0],
                embedding_dim=module.weight.size()[1],
                orig_module=None,
            )
        else:
            tensor_list = [
                torch.zeros_like(module.weight.data) for _ in range(world_size)
            ]
            dist.all_gather(
                tensor_list,
                module.weight.data.contiguous(),
                self.parallel_context.get_group(ParallelMode.TENSOR_1D),
            )
            w = torch.cat(tensor_list, dim=1)
            module.weight.data = w

            _update_module_arguments(
                module=module,
                parallel_context=None,
                embedding_dim=module.weight.size()[1],
            )
        module.__class__ = nn.Embedding

    def _gather_linear(self, module, dim=1):
        is_reversed = module.reversed
        fusion_degree = module.fusion_degree

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)

        w = self._reconstruct_combined_qkv(
            module.weight, world_size, fusion_degree, dim
        )
        if is_reversed:
            w = w.t()
        module.weight.data = w

        if hasattr(module, "bias") and module.bias is not None and dim != 1:
            b = self._reconstruct_combined_qkv(
                module.bias, world_size, fusion_degree, dim
            )
            module.bias.data = b

        _update_module_arguments(
            module=module,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_context=self.parallel_context,
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
        )

        del module.reversed
        del module.fusion_degree
        del module.orig_module
        del module.parallel_context
        module.__class__ = nn.Linear

    def _gather_column_linear(self, module):
        self._gather_linear(module, dim=0)

    def _gather_row_linear(self, module):
        self._gather_linear(module, dim=1)

    def _reconstruct_combined_qkv(self, tensor, world_size, fusion_degree, dim: int):
        tensor_list = tensor.chunk(fusion_degree, dim=dim)
        result_list = []
        for w in tensor_list:
            w_list = [torch.zeros_like(w) for _ in range(world_size)]
            dist.all_gather(
                w_list, w, self.parallel_context.get_group(ParallelMode.TENSOR_1D)
            )
            result_list.append(torch.cat(w_list, dim=dim))
        return torch.cat(result_list, dim=dim)
