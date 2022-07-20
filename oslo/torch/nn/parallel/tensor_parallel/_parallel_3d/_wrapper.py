import copy

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding3D,
    Embedding3D,
    VocabUtility,
)
from oslo.torch.nn.modules.linear import (
    Linear3D,
)
from oslo.torch.nn.modules.layer_norm import (
    LayerNorm3D,
)
from oslo.torch.distributed.nn.functional import (
    scatter,
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


class _TensorParallel3D(ParallelWrapper):
    """
    PyTorch module for 3D tensor parallelism

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
        assert len(args) == 0, (
            "3D tensor parallel model only supports ``**kwargs`` input (keyword arguments). "
            "If you wrote code like ``model(input_ids, labels)``, "
            "please modify your code like ``model(input_ids=input_ids, labels=labels)``."
        )
        if not is_oslo_model(self.module):
            kwargs = {
                key: scatter(
                    scatter(
                        value,
                        dim=BATCH_DIMENSIONS[key],
                        parallel_context=self.parallel_context,
                        parallel_mode=ParallelMode.TENSOR_3D_WEIGHT,
                    ),
                    dim=BATCH_DIMENSIONS[key],
                    parallel_context=self.parallel_context,
                    parallel_mode=ParallelMode.TENSOR_3D_INPUT,
                )
                if key in BATCH_DIMENSIONS
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
        self._parallelize_head()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    def _update_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    cubic_dim = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_3D_INPUT
                    )
                    assert (
                        getattr(module, elem.name) % cubic_dim == 0
                    ), f"{elem.name} ({getattr(module, elem.name)}) must be divisible by cubic_dim ({cubic_dim})."
                    reduced_arg = getattr(module, elem.name) // cubic_dim
                    setattr(module, elem.name, reduced_arg)

    def _parallelize_embedding(self):
        for module in self.module.modules():
            if isinstance(module, nn.Embedding):
                self._slice_embedding(
                    module=module,
                )

    def _parallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if self.tensor_parallel_mapping.is_column_parallel(
                self.module, param_name
            ) or self.tensor_parallel_mapping.is_row_parallel(self.module, param_name):
                self._slice_linear(
                    module=module,
                    reversed=self.tensor_parallel_mapping.is_reversed(
                        self.module, param_name
                    ),
                    fusion_degree=self.tensor_parallel_mapping.get_combined_qkv_degree(
                        self.module, param_name, module
                    ),
                    slice_bias=True,
                )

    def _parallelize_layernorm(self):
        for module in self.module.modules():
            if isinstance(module, nn.LayerNorm):
                self._slice_layernorm(
                    module=module,
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
    def _deconstruct_combined_qkv(tensor, cubic_dim, fusion_degree, is_bias=False):
        if is_bias:
            tensor = [
                [tensor[j * cubic_dim + k] for k in range(cubic_dim)]
                for j in range(fusion_degree)
            ]
            tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
            tensor = [tensor[j] for j in range(cubic_dim)]
        else:
            tensor = [
                [
                    tensor[i][j * cubic_dim + k]
                    for i in range(cubic_dim)
                    for k in range(cubic_dim)
                ]
                for j in range(fusion_degree)
            ]
            tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
            tensor = [
                [tensor[i * cubic_dim + j] for j in range(cubic_dim)]
                for i in range(cubic_dim)
            ]
        return tensor

    def _slice_embedding(self, module):
        cubic_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
        input_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
        output_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_OUTPUT
        )
        weight_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_WEIGHT
        )

        if module is self.module.get_input_embeddings():
            (
                vocab_start_index,
                vocab_end_index,
            ) = VocabUtility.vocab_range_from_global_vocab_size(
                module.num_embeddings,
                input_rank,
                cubic_dim,
            )

            weight_list = module.weight.data.chunk(cubic_dim, dim=1)
            weight_list = [weight.chunk(cubic_dim, dim=0) for weight in weight_list]
            weight_list = [
                [weight.chunk(cubic_dim, dim=0) for weight in weights]
                for weights in weight_list
            ]

            module.weight.data = weight_list[output_rank][input_rank][
                weight_rank
            ].contiguous()

            _update_module_arguments(
                module=module,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                parallel_context=self.parallel_context,
                cubic_dim=cubic_dim,
                num_embeddings=module.weight.size()[0],
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = VocabParallelEmbedding3D
        else:
            weight = module.weight.data.chunk(cubic_dim, dim=-1)
            module.weight.data = weight[output_rank].contiguous()

            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                cubic_dim=cubic_dim,
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = Embedding3D

        if hasattr(module.weight, "oslo_parallel"):
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_INPUT] = input_rank
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_OUTPUT] = output_rank
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_WEIGHT] = weight_rank
        else:
            module.weight.oslo_parallel = {
                ParallelMode.TENSOR_3D_INPUT: input_rank,
                ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
            }

    def _slice_linear(self, module, reversed, fusion_degree, slice_bias):
        cubic_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
        input_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
        output_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_OUTPUT
        )
        weight_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_WEIGHT
        )

        if reversed:
            module.weight.data = module.weight.data.t()

        weight_list = module.weight.data.chunk(cubic_dim, dim=1)
        weight_list = [
            weight.chunk(fusion_degree * cubic_dim, dim=0) for weight in weight_list
        ]

        if fusion_degree > 1:
            weight_list = self._deconstruct_combined_qkv(
                weight_list,
                cubic_dim,
                fusion_degree,
                is_bias=False,
            )
        weight_list = [
            [weight.chunk(cubic_dim, dim=0) for weight in weights]
            for weights in weight_list
        ]

        module.weight.data = weight_list[output_rank][input_rank][
            weight_rank
        ].contiguous()

        if hasattr(module.weight, "oslo_parallel"):
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_INPUT] = input_rank
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_OUTPUT] = output_rank
            module.weight.oslo_parallel[ParallelMode.TENSOR_3D_WEIGHT] = weight_rank
        else:
            module.weight.oslo_parallel = {
                ParallelMode.TENSOR_3D_INPUT: input_rank,
                ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
            }

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.data.chunk(fusion_degree * cubic_dim, dim=0)

                if fusion_degree > 1:
                    bias_list = self._deconstruct_combined_qkv(
                        bias_list,
                        cubic_dim,
                        fusion_degree,
                        is_bias=True,
                    )

                module.bias.data = bias_list[input_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_3D_INPUT] = input_rank
                    module.bias.oslo_parallel[
                        ParallelMode.TENSOR_3D_OUTPUT
                    ] = output_rank
                    module.bias.oslo_parallel[
                        ParallelMode.TENSOR_3D_WEIGHT
                    ] = weight_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_3D_INPUT: input_rank,
                        ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                        ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
                    }

        _update_module_arguments(
            module=module,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
            parallel_context=self.parallel_context,
            cubic_dim=cubic_dim,
            reversed=reversed,
            fusion_degree=fusion_degree,
            skip_bias_add=module.skip_bias_add
            if hasattr(module, "skip_bias_add")
            else False,
            gather_output=False,
            orig_module=copy.deepcopy(module.__class__),
        )

        module.__class__ = Linear3D

    def _slice_layernorm(self, module):
        cubic_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
        input_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
        output_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_OUTPUT
        )
        weight_rank = self.parallel_context.get_local_rank(
            ParallelMode.TENSOR_3D_WEIGHT
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                weight_list = module.weight.data.chunk(cubic_dim, dim=0)
                module.weight.data = weight_list[output_rank].contiguous()

                if hasattr(module.weight, "oslo_parallel"):
                    module.weight.oslo_parallel[
                        ParallelMode.TENSOR_3D_INPUT
                    ] = input_rank
                    module.weight.oslo_parallel[
                        ParallelMode.TENSOR_3D_OUTPUT
                    ] = output_rank
                    module.weight.oslo_parallel[
                        ParallelMode.TENSOR_3D_WEIGHT
                    ] = weight_rank
                else:
                    module.weight.oslo_parallel = {
                        ParallelMode.TENSOR_3D_INPUT: input_rank,
                        ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                        ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
                    }

        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.dim() >= 1:
                bias_list = module.bias.chunk(cubic_dim, dim=0)
                module.bias.data = bias_list[input_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_3D_INPUT] = input_rank
                    module.bias.oslo_parallel[
                        ParallelMode.TENSOR_3D_OUTPUT
                    ] = output_rank
                    module.bias.oslo_parallel[
                        ParallelMode.TENSOR_3D_WEIGHT
                    ] = weight_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_3D_INPUT: input_rank,
                        ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                        ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
                    }

        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0] * cubic_dim,
            partitioned_dim=module.weight.size()[0],
            parallel_context=self.parallel_context,
            cubic_dim=cubic_dim,
            orig_module=copy.deepcopy(module.__class__),
        )
        module.__class__ = LayerNorm3D

    def _slice_head(self, module, reversed):
        if module.weight is not self.module.get_input_embeddings().weight:
            self._slice_linear(
                module=module,
                reversed=reversed,
                fusion_degree=1,
                slice_bias=True,
            )
            _update_module_arguments(
                module=module,
                gather_output=not is_oslo_model(self.module),
            )
        else:
            cubic_dim = self.parallel_context.get_world_size(
                ParallelMode.TENSOR_3D_INPUT
            )
            input_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
            output_rank = self.parallel_context.get_local_rank(
                ParallelMode.TENSOR_3D_OUTPUT
            )
            weight_rank = self.parallel_context.get_local_rank(
                ParallelMode.TENSOR_3D_WEIGHT
            )

            if hasattr(module, "bias") and module.bias is not None:
                if module.bias.dim() >= 1:
                    bias_list = module.bias.data.chunk(cubic_dim, dim=0)
                    module.bias.data = bias_list[input_rank].contiguous()

                    if hasattr(module.bias, "oslo_parallel"):
                        module.bias.oslo_parallel[ParallelMode.TENSOR_3D_INPUT] = input_rank
                        module.bias.oslo_parallel[
                            ParallelMode.TENSOR_3D_OUTPUT
                        ] = output_rank
                        module.bias.oslo_parallel[
                            ParallelMode.TENSOR_3D_WEIGHT
                        ] = weight_rank
                    else:
                        module.bias.oslo_parallel = {
                            ParallelMode.TENSOR_3D_INPUT: input_rank,
                            ParallelMode.TENSOR_3D_OUTPUT: output_rank,
                            ParallelMode.TENSOR_3D_WEIGHT: weight_rank,
                        }

            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[1],
                out_features=module.weight.size()[0],
                parallel_context=self.parallel_context,
                cubic_dim=cubic_dim,
                reversed=reversed,
                fusion_degree=1,
                skip_bias_add=False,
                gather_output=not is_oslo_model(self.module),
                orig_module=copy.deepcopy(module.__class__),
            )
        module.__class__ = Linear3D
