import copy
from re import L

import torch
import torch.nn as nn
from torch.nn import Embedding

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding2D,
    Embedding2D,
    VocabUtility,
)
from oslo.torch.nn.modules.linear import (
    Linear2D,
)
from oslo.torch.nn.modules.layer_norm import (
    LayerNorm2D,
)
from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
    split_batch_2d,
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

from oslo.transformers.constants import BATCH_DIMENSIONS


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

    def forward(self, *args, **kwargs):
        if self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
            assert len(args) == 0, (
                "2D tensor parallel model only supports ``**kwargs`` input (keyword arguments). "
                "If you wrote code like ``model(input_ids, labels)``, "
                "please modify your code like ``model(input_ids=input_ids, labels=labels)``."
            )
            if not is_oslo_model(self.module):
                kwargs = {
                    key: split_batch_2d(
                        value,
                        dim=BATCH_DIMENSIONS[key],
                        parallel_context=self.parallel_context,
                    )
                    if value in BATCH_DIMENSIONS
                    else value
                    for key, value in kwargs.items()
                }
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def _parallelize(self):
        self._update_mp_arguments()
        self._parallelize_embedding()
        self._parallelize_linear()
        self._parallelize_layernorm()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    def _update_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    summa_dim = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_2D_COL
                    )
                    reduced_arg = getattr(module, elem.name) // summa_dim
                    setattr(module, elem.name, reduced_arg)

    @staticmethod
    def _deconstruct_combined_qkv(tensor, summa_dim, fusion_degree):
        tensor = [
            [
                tensor[i][j * summa_dim + k]
                for i in range(summa_dim)
                for k in range(summa_dim)
            ]
            for j in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
        tensor = [
            [tensor[i * summa_dim + j] for j in range(summa_dim)]
            for i in range(summa_dim)
        ]
        return tensor

    def _slice_linear(self, module, reversed, fusion_degree, slice_bias):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        summa_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)

        data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            row_rank=row_rank,
            col_rank=col_rank,
            summa_dim=summa_dim,
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            reversed=reversed,
            fusion_degree=fusion_degree,
            orig_module=copy.deepcopy(module.__class__),
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
            gather_output=False,
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                if reversed:
                    module.weight.data = module.weight.data.t()

                weight_list = module.weight.data.chunk(summa_dim, dim=1)
                weight_list = [
                    weight.chunk(fusion_degree * summa_dim, dim=0)
                    for weight in weight_list
                ]

                if fusion_degree > 1:
                    weight_list = self._deconstruct_combined_qkv(
                        weight_list,
                        summa_dim,
                        fusion_degree,
                    )

                module.weight.data = weight_list[row_rank][col_rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_2D_ROW: row_rank,
                        ParallelMode.TENSOR_2D_COL: col_rank,
                    }

            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[1],
                out_features=module.weight.size()[0],
            )

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.chunk(summa_dim, dim=0)
                bias_list = [
                    bias.chunk(fusion_degree * summa_dim, dim=0) for bias in bias_list
                ]

                if fusion_degree > 1:
                    bias_list = self._deconstruct_combined_qkv(
                        bias_list,
                        summa_dim,
                        fusion_degree,
                    )

                module.bias.data = bias_list[row_rank][col_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2D_ROW: row_rank,
                        ParallelMode.TENSOR_2D_COL: col_rank,
                    }

        return module

    def _slice_layernorm(self, module):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        summa_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)

        data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            row_rank=row_rank,
            col_rank=col_rank,
            summa_dim=summa_dim,
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            orig_module=copy.deepcopy(module.__class__),
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                weight_list = module.weight.data.chunk(summa_dim, dim=0)
                weight_list = [weight.chunk(summa_dim, dim=0) for weight in weight_list]
                module.weight.data = weight_list[row_rank][col_rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_2D_ROW: row_rank,
                        ParallelMode.TENSOR_2D_COL: col_rank,
                    }

        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.dim() >= 1:
                bias_list = module.bias.chunk(summa_dim, dim=0)
                bias_list = [bias.chunk(summa_dim, dim=0) for bias in bias_list]
                module.bias.data = bias_list[row_rank][col_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2D_ROW: row_rank,
                        ParallelMode.TENSOR_2D_COL: col_rank,
                    }

        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0] * (summa_dim**2),
            partitioned_dim=module.weight.size()[0],
        )

        return module

    def _slice_embedding(self, module):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        summa_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)

        if module is self.module.get_input_embeddings():
            data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
            pipeline_parallel_rank = self.parallel_context.get_local_rank(
                ParallelMode.PIPELINE
            )
            tensor_parallel_size = self.parallel_context.get_world_size(
                ParallelMode.TENSOR
            )
            pipeline_parallel_size = self.parallel_context.get_world_size(
                ParallelMode.PIPELINE
            )

            (
                vocab_start_index,
                vocab_end_index,
            ) = VocabUtility.vocab_range_from_global_vocab_size(
                module.num_embeddings,
                col_rank,
                summa_dim,
            )

            weight_list = module.weight.data.chunk(summa_dim, dim=1)
            weight_list = [weight.chunk(summa_dim, dim=0) for weight in weight_list]

            module.weight.data = weight_list[row_rank][col_rank].contiguous()

            if hasattr(module.weight, "oslo_parallel"):
                module.weight.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                module.weight.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
            else:
                module.weight.oslo_parallel = {
                    ParallelMode.TENSOR_2D_ROW: row_rank,
                    ParallelMode.TENSOR_2D_COL: col_rank,
                }

            _update_module_arguments(
                module=module,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                parallel_context=self.parallel_context,
                num_embeddings=module.weight.size()[0],
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )

            if isinstance(module, Embedding):
                module.__class__ = VocabParallelEmbedding2D

            for name, _module in self.module.named_modules():
                if (
                    hasattr(_module, "weight")
                    and _module.weight is module.weight
                    and not isinstance(_module, Embedding)
                ):
                    _update_module_arguments(
                        module=_module,
                        parallel_context=self.parallel_context,
                        row_rank=row_rank,
                        col_rank=col_rank,
                        summa_dim=summa_dim,
                        data_parallel_rank=data_parallel_rank,
                        pipeline_parallel_rank=pipeline_parallel_rank,
                        tensor_parallel_size=tensor_parallel_size,
                        pipeline_parallel_size=pipeline_parallel_size,
                        reversed=self.tensor_parallel_mapping.is_reversed_param(
                            self.module, name
                        ),
                        fusion_degree=1,
                        orig_module=copy.deepcopy(_module.__class__),
                        gather_output=not is_oslo_model(self.module),
                        in_features=module.weight.size()[1],
                        out_features=module.weight.size()[0],
                    )

                    if isinstance(_module, nn.Linear):
                        _module.__class__ = Linear2D
                    else:
                        raise RuntimeError("Classifier layer must be `nn.Linear` class")
        else:
            weight_list = module.weight.data.chunk(summa_dim, dim=1)
            weight_list = [weight.chunk(summa_dim, dim=1) for weight in weight_list]
            module.weight.data = weight_list[row_rank][col_rank].contiguous()

            if hasattr(module.weight, "oslo_parallel"):
                module.weight.oslo_parallel[ParallelMode.TENSOR_2D_ROW] = row_rank
                module.weight.oslo_parallel[ParallelMode.TENSOR_2D_COL] = col_rank
            else:
                module.weight.oslo_parallel = {
                    ParallelMode.TENSOR_2D_ROW: row_rank,
                    ParallelMode.TENSOR_2D_COL: col_rank,
                }

            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )

            if isinstance(module, Embedding):
                module.__class__ = Embedding2D

    def _parallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(
                self.module, param_name
            ) or self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                self._slice_linear(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed_param(
                        self.module, param_name
                    ),
                    fusion_degree=self.tensor_parallel_mapping.get_combined_qkv_degree(
                        self.module, param_name, module
                    ),
                    slice_bias=True,
                )
                module.__class__ = Linear2D

    def _parallelize_layernorm(self):
        for module in self.module.modules():
            if isinstance(module, nn.LayerNorm):
                self._slice_layernorm(
                    module=module,
                )
                module.__class__ = LayerNorm2D

    def _parallelize_embedding(self):
        for module in self.module.modules():
            if isinstance(module, nn.Embedding):
                self._slice_embedding(
                    module=module,
                )
