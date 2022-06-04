from typing import Dict

import torch
import torch.nn as nn

import copy

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel.mapping import (
    TensorParallelMapping,
)
from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding2p5D,
    VocabUtility,
    Embedding2p5D
)
from oslo.torch.nn.modules.lazy import LazyModuleMixin
from oslo.torch.nn.modules.linear import (
    Linear2p5D
)
from oslo.torch.nn.modules.layer_norm import LayerNorm2p5D
from oslo.transformers.mapping_utils import (
    _TensorParallelMappingForHuggingFace,
)
from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    _update_module_arguments,
    is_huggingface_model,
    is_oslo_model,
)


class _TensorParallel2p5D(ParallelWrapper):
    """
    PyTorch module for 2.5D tensor parallelism

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
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def _parallelize(self):
        self._update_mp_arguments()
        self._parallelize_embedding()
        self._parallalize_linear()
        self._parallelize_layernorm()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    # TODO: erase?
    def _update_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    tesseract_dim = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_2P5D_COL
                    )
                    reduced_arg = getattr(module, elem.name) // tesseract_dim
                    setattr(module, elem.name, reduced_arg)

    def _parallalize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(self.module, param_name) \
                    or self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
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
                module.__class__ = Linear2p5D

    def _parallelize_layernorm(self):
        for module in self.module.modules():
            if isinstance(module, nn.LayerNorm):
                self._slice_layernorm(
                    module=module,
                )
                module.__class__ = LayerNorm2p5D

    def _parallelize_embedding(self):
        for module in self.module.modules():
            if isinstance(module, nn.Embedding):
                self._slice_embedding(
                    module=module,
                )

    @staticmethod
    def _deconstruct_combined_qkv(tensor, tessearct_dim, fusion_degree, is_bias=False):
        tensor = [
            [
                tensor[i][j * tessearct_dim + k]
                for i in range(tessearct_dim)
                for k in range(tessearct_dim)
            ]
            for j in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
        tensor = [
            [tensor[i * tessearct_dim + j] for j in range(tessearct_dim)]
            for i in range(tessearct_dim)
        ]
        return tensor

    @staticmethod
    def _deconstrunct_combined_qkv_bias(tensor, tessearct_dim, fusion_degree):
        tensor = [
            [
                tensor[j * tessearct_dim + k]
                for k in range(tessearct_dim)
            ]
            for j in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
        tensor = [tensor[j] for j in range(tessearct_dim)]
        return tensor

    def _slice_linear(self, module, reversed, fusion_degree, slice_bias):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

        data_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.DATA
        )
        pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.TENSOR
        )
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            row_rank=row_rank,
            col_rank=col_rank,
            dep_rank=dep_rank,
            tesseract_dim=tesseract_dim,
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
                if isinstance(module, LazyModuleMixin):
                    module.initialize_parameters()

                if reversed:
                    module.weight.data = module.weight.data.t()

                weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
                weight_list = [weight.chunk(tesseract_dim * fusion_degree, dim=0) for weight in weight_list]

                if fusion_degree > 1:
                    weight_list = self._deconstruct_combined_qkv(
                        weight_list, tesseract_dim, fusion_degree,
                    )

                if isinstance(module, LazyModuleMixin):
                    new_tensor = weight_list[row_rank][col_rank].clone()
                    del weight_list, module.weight
                    module.weight = nn.Parameter(new_tensor.contiguous())
                else:
                    module.weight.data = weight_list[row_rank][col_rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                        ParallelMode.TENSOR_2P5D_COL: col_rank,
                    }

            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[1],
                out_features=module.weight.size()[0],
            )

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.chunk(tesseract_dim * fusion_degree, dim=0)
                # bias_list = [bias.chunk(1 * fusion_degree, dim=0) for bias in bias_list]

                if fusion_degree > 1:
                    bias_list = self._deconstrunct_combined_qkv_bias(
                        bias_list, tesseract_dim, fusion_degree
                    )

                if isinstance(module, LazyModuleMixin):
                    new_tensor = bias_list[row_rank].clone()
                    del bias_list, module.bias
                    module.bias = nn.Parameter(new_tensor.contiguous())
                else:
                    module.bias.data = bias_list[row_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                        ParallelMode.TENSOR_2P5D_COL: col_rank,
                    }

        return module

    def _slice_layernorm(self, module):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

        data_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.DATA
        )
        pipeline_parallel_rank=self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.TENSOR
        )
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            row_rank=row_rank,
            col_rank=col_rank,
            dep_rank=dep_rank,
            tesseract_dim=tesseract_dim,
            data_parallel_rank=data_parallel_rank,
            pipeline_parallel_rank=pipeline_parallel_rank,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            orig_module=copy.deepcopy(module.__class__),
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                weight_list = module.weight.data.chunk(tesseract_dim, dim=0)
                module.weight.data = weight_list[row_rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                    }

        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.dim() >= 1:
                bias_list = module.bias.chunk(tesseract_dim, dim=0)
                module.bias.data = bias_list[row_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                    }

        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0] * tesseract_dim,
            partitioned_dim=module.weight.size()[0],
        )

        return module

    def _slice_embedding(self, module):
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)

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
                module.num_embeddings, col_rank, tesseract_dim
            )

            if isinstance(module, LazyModuleMixin):
                assert hasattr(module, "weight"), "embedding must has `weight`."
                module.initialize_parameters()

            # w = module.weight.data.chunk(tesseract_dim, dim=1)[
            #     self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
            # ]
            # w = w.chunk(tesseract_dim, dim=1)[
            #     self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
            # ]

            weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
            weight_list = [weight.chunk(tesseract_dim, dim=0) for weight in weight_list]

            if isinstance(module, LazyModuleMixin):
                new_tensor = weight_list[row_rank][col_rank].clone()
                del weight_list, module.weight
                module.weight = nn.Parameter(new_tensor.contiguous())
            else:
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

            if isinstance(module, nn.Embedding):
                module.__class__ = VocabParallelEmbedding2p5D

            for name, _module in self.module.named_modules():
                if (
                        hasattr(_module, "weight")
                        and _module.weight is module.weight
                        and not isinstance(_module, nn.Embedding)
                ):
                    _update_module_arguments(
                        module=_module,
                        parallel_context=self.parallel_context,
                        row_rank=row_rank,
                        col_rank=col_rank,
                        dep_rank=dep_rank,
                        tesseract_dim=tesseract_dim,
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
                        _module.__class__ = Linear2p5D
                    else:
                        raise RuntimeError("Classifier layer must be `nn.Linear` class")
        else:
            weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
            weight_list = [weight.chunk(tesseract_dim, dim=1) for weight in weight_list]
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

            if isinstance(module, nn.Embedding):
                module.__class__ = Embedding2p5D

        # while isinstance(module, ParallelWrapper):
        #     module = module.module
        #
        # assert hasattr(module, "get_input_embeddings"), (
        #     "model object must have `get_input_embeddings` and "
        #     "`get_output_embeddings` method."
        # )
        #
        # # world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        # tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
        # row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        # col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        # embedding = module.get_input_embeddings()
        #
        # data_parallel_rank = self.parallel_context.get_local_rank(
        #     ParallelMode.DATA
        # )
        # pipeline_parallel_rank=self.parallel_context.get_local_rank(
        #     ParallelMode.PIPELINE
        # )
        # tensor_parallel_size = self.parallel_context.get_world_size(
        #     ParallelMode.TENSOR
        # )
        # pipeline_parallel_size = self.parallel_context.get_world_size(
        #     ParallelMode.PIPELINE
        # )
        #
        # (
        #     vocab_start_index,
        #     vocab_end_index,
        # ) = VocabUtility.vocab_range_from_global_vocab_size(
        #     embedding.num_embeddings, col_rank, tesseract_dim
        # )
        # if isinstance(embedding, LazyModuleMixin):
        #     assert hasattr(embedding, "weight"), "embedding must has `weight`."
        #     embedding.initialize_parameters()
        #
        # w = embedding.weight.data.chunk(tesseract_dim, dim=1)[
        #     self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        # ]
        # w = w.chunk(tesseract_dim, dim=1)[
        #     self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        # ]
        #
        # if isinstance(embedding, LazyModuleMixin):
        #     new_tensor = w.clone()
        #     del w, embedding.weight
        #     embedding.weight = nn.Parameter(new_tensor.contiguous())
        # else:
        #     embedding.weight.data = w.contiguous()
        #
        #     if hasattr(embedding.weight, "oslo_parallel"):
        #         embedding.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
        #         embedding.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
        #     else:
        #         embedding.weight.oslo_parallel = {
        #             ParallelMode.TENSOR_2P5D_ROW: row_rank,
        #             ParallelMode.TENSOR_2P5D_COL: col_rank,
        #         }
        #
        # _update_module_arguments(
        #     module=embedding,
        #     vocab_start_index=vocab_start_index,
        #     vocab_end_index=vocab_end_index,
        #     parallel_context=self.parallel_context,
        #     num_embeddings=embedding.weight.size()[0],
        #     embedding_dim=embedding.weight.size()[1],
        #     orig_module=copy.deepcopy(embedding.__class__),
        # )
        #
        # if isinstance(embedding, nn.Embedding):
        #     embedding.__class__ = VocabParallelEmbedding2p5D
        #
        # for name, _module in module.named_modules():
        #     if (
        #             hasattr(_module, "weight")
        #             and _module.weight is embedding.weight
        #             and not isinstance(_module, nn.Embedding)
        #     ):
        #         _update_module_arguments(
        #             module=_module,
        #             parallel_context=self.parallel_context,
        #             row_rank=row_rank,
        #             col_rank=col_rank,
        #             tesseract_dim=tesseract_dim,
        #             data_parallel_rank=data_parallel_rank,
        #             pipeline_parallel_rank=pipeline_parallel_rank,
        #             tensor_parallel_size=tensor_parallel_size,
        #             pipeline_parallel_size=pipeline_parallel_size,
        #             reversed=self.tensor_parallel_mapping.is_reversed_param(
        #                 self.module, name
        #             ),
        #             fusion_degree=1,
        #             orig_module=copy.deepcopy(_module.__class__),
        #             out_features=embedding.weight.size()[0],
        #         )
        #
        #         # Embedding : 1d parallel, Linear : 2.5d parallel
        #         if isinstance(_module, nn.Linear):
        #             _module.__class__ = Linear2p5D
        #         else:
        #             raise RuntimeError("Classifier layer must be `nn.Linear` class")
        #
