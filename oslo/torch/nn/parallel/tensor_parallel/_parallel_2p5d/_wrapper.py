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
)
from oslo.torch.nn.modules.lazy import LazyModuleMixin
from oslo.torch.nn.modules.linear import (
    Linear2p5D,
)
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

    @torch.no_grad()
    def _parallelize(self):
        # self._update_mp_arguments()
        self._parallelize_embedding()
        self._parallelize_modules()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    # TODO: erase?
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
        module = self.module
        while isinstance(module, ParallelWrapper):
            module = module.module

        assert hasattr(module, "get_input_embeddings"), (
            "model object must have `get_input_embeddings` and "
            "`get_output_embeddings` method."
        )

        # world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        embedding = module.get_input_embeddings()

        (
            vocab_start_index,
            vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            embedding.num_embeddings, col_rank, tesseract_dim
        )
        if isinstance(embedding, LazyModuleMixin):
            assert hasattr(embedding, "weight"), "embedding must has `weight`."
            embedding.initialize_parameters()

        # linear - 2p5d, embedding - 1d
        # split weight into 0,4:[0], 1,5:[2], 2,6:[1], 3,7:[3]
        w = embedding.weight.data.chunk(tesseract_dim, dim=1)[
            self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        ]
        w = w.chunk(tesseract_dim, dim=1)[
            self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        ]

        if isinstance(embedding, LazyModuleMixin):
            new_tensor = w.clone()
            del w, embedding.weight
            embedding.weight = nn.Parameter(new_tensor.contiguous())
        else:
            embedding.weight.data = w.contiguous()

            if hasattr(embedding.weight, "oslo_parallel"):
                embedding.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                embedding.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
            else:
                embedding.weight.oslo_parallel = {
                    ParallelMode.TENSOR_2P5D_ROW: row_rank,
                    ParallelMode.TENSOR_2P5D_COL: col_rank,
                }

        _update_module_arguments(
            module=embedding,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
            parallel_context=self.parallel_context,
            num_embeddings=embedding.weight.size()[0],
            orig_module=copy.deepcopy(embedding.__class__),
        )

        if isinstance(embedding, nn.Embedding):
            embedding.__class__ = VocabParallelEmbedding2p5D

        for name, _module in module.named_modules():
            if (
                    hasattr(_module, "weight")
                    and _module.weight is embedding.weight
                    and not isinstance(_module, nn.Embedding)
            ):
                _update_module_arguments(
                    module=_module,
                    parallel_context=self.parallel_context,
                    fusion_degree=1,
                    orig_module=copy.deepcopy(_module.__class__),
                    out_features=embedding.weight.size()[0],
                )

                # Embedding : 1d parallel, Linear : 2.5d parallel
                if isinstance(_module, nn.Linear):
                    _module.__class__ = Linear2p5D
                else:
                    raise RuntimeError("Classifier layer must be `nn.Linear` class")

    def _parallelize_modules(self):
        for param_name, module in self.module.named_modules():
            self._slice(
                module=module,
                reversed=self.tensor_parallel_mapping.is_reversed_param(
                    self.module, param_name
                ),
                fusion_degree=self.tensor_parallel_mapping.get_combined_qkv_degree(
                    self.module, param_name, module
                ),
                slice_bias=True
            )
            module.__class__ = Linear2p5D

    def _slice(self, module, reversed, fusion_degree, slice_bias):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)

        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

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
                if isinstance(module, LazyModuleMixin):
                    module.initialize_parameters()

                # TODO: how to deal with fusion degree?
                # if fusion_degree > 1:
                #     weight_list = self._deconstruct_combined_qkv(
                #         weight_list, world_size
                #     )

                # split weight into 0,4:[0, 0], 1,5:[1, 0], 2,6:[0, 1], 3,7:[1, 1]
                # input shape: (n/q, k/q)
                w = module.weight.chunk(tesseract_dim, dim=1)[
                    self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
                ]
                w = w.chunk(tesseract_dim, dim=0)[
                    self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
                ]

                if isinstance(module, LazyModuleMixin):
                    new_tensor = w.clone()
                    del w, module.weight
                    module.weight = nn.Parameter(new_tensor.contiguous())
                else:
                    module.weight.data = w.contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_DEP] = dep_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                        ParallelMode.TENSOR_2P5D_COL: col_rank,
                        ParallelMode.TENSOR_2P5D_DEP: dep_rank,
                    }

            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[0],
                out_features=module.weight.size()[1],
            )

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                # TODO: how to deal with fusion degree?
                # bias_list = module.bias.chunk(fusion_degree * world_size, dim=0)
                # split bias into 0,4:[0], 2,6:[1]
                # input shape: (k/q)
                b = module.bias.chunk(tesseract_dim, dim=0)[
                    self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
                ]
                # if fusion_degree > 1:
                #     bias_list = self._deconstruct_combined_qkv(bias_list, world_size)

                if isinstance(module, LazyModuleMixin):
                    new_tensor = b.clone()
                    del b, module.bias
                    module.bias = nn.Parameter(new_tensor.contiguous())
                else:
                    module.bias.data = b.contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                else:
                    module.bias.oslo_parallel = {ParallelMode.TENSOR_2P5D_ROW: row_rank}

        return module
