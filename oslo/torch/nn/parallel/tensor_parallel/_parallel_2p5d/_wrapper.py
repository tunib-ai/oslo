import copy

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.nn.modules.embedding import (
    VocabParallelEmbedding2p5D,
    VocabUtility,
    Embedding2p5D,
)
from oslo.torch.nn.modules.linear import Linear2p5D
from oslo.torch.nn.modules.layer_norm import LayerNorm2p5D
from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
    split_batch_2p5d,
    gather_2d,
    gather_1d
)

from oslo.torch.nn.parallel.tensor_parallel.mapping import (
    TensorParallelMapping,
)
from oslo.torch.nn.parallel.tensor_parallel._base_wrapper import (
    BaseTensorParallelWrapper,
)

from oslo.torch.nn.parallel.utils import (
    _update_module_arguments,
    is_huggingface_model,
    is_oslo_model
)
from oslo.transformers.mapping_utils import (
    _TensorParallelMappingForHuggingFace,
)

from oslo.transformers.constants import BATCH_DIMENSIONS


class _TensorParallel2p5D(BaseTensorParallelWrapper):
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
            module_args: dict = None
    ):
        super().__init__(module, parallel_context, mapping, module_args)
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
        assert len(args) == 0, (
            "2.5D tensor parallel model only supports ``**kwargs`` input (keyword arguments). "
            "If you wrote code like ``model(input_ids, labels)``, "
            "please modify your code like ``model(input_ids=input_ids, labels=labels)``."
        )
        if not is_oslo_model(self.module):
            kwargs = {
                key: split_batch_2p5d(
                    value,
                    dim=BATCH_DIMENSIONS[key],
                    parallel_context=self.parallel_context,
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
        self._parallalize_linear()
        self._parallelize_layernorm()
        self._parallelize_head()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    def _update_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    tesseract_dim = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_2P5D_COL
                    )
                    assert (
                            getattr(module, elem.name) % tesseract_dim == 0
                    ), f"{elem.name} must be divisible by tesseract_dim."
                    reduced_arg = getattr(module, elem.name) // tesseract_dim
                    setattr(module, elem.name, reduced_arg)

    def _parallelize_embedding(self):
        for module in self.module.modules():
            if isinstance(module, nn.Embedding):
                self._slice_embedding(
                    module=module,
                )

    def _parallalize_linear(self):
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
    def _deconstruct_combined_qkv(tensor, tessearct_dim, fusion_degree, is_bias=False):
        if is_bias:
            tensor = [
                [tensor[j * tessearct_dim + k] for k in range(tessearct_dim)]
                for j in range(fusion_degree)
            ]
            tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
            tensor = [tensor[j] for j in range(tessearct_dim)]
        else:
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

    def _slice_embedding(self, module):
        tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)

        if module is self.module.get_input_embeddings():
            (
                vocab_start_index,
                vocab_end_index,
            ) = VocabUtility.vocab_range_from_global_vocab_size(
                module.num_embeddings, col_rank, tesseract_dim
            )

            weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
            weight_list = [weight.chunk(tesseract_dim, dim=0) for weight in weight_list]

            module.weight.data = weight_list[row_rank][col_rank].contiguous()

            _update_module_arguments(
                module=module,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                parallel_context=self.parallel_context,
                num_embeddings=module.weight.size()[0],
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = VocabParallelEmbedding2p5D
        else:
            weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
            weight_list = [weight.chunk(tesseract_dim, dim=1) for weight in weight_list]
            module.weight.data = weight_list[row_rank][col_rank].contiguous()

            _update_module_arguments(
                module=module,
                parallel_context=self.parallel_context,
                embedding_dim=module.weight.size()[1],
                orig_module=copy.deepcopy(module.__class__),
            )
            module.__class__ = Embedding2p5D

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

    def _slice_linear(self, module, reversed, fusion_degree, slice_bias):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
        tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )

        data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        if reversed:
            module.weight.data = module.weight.data.t()

        weight_list = module.weight.data.chunk(tesseract_dim, dim=1)
        weight_list = [
            weight.chunk(tesseract_dim * fusion_degree, dim=0) for weight in weight_list
        ]

        if fusion_degree > 1:
            weight_list = self._deconstruct_combined_qkv(
                weight_list,
                tesseract_dim,
                fusion_degree,
                is_bias=False,
            )

        module.weight.data = weight_list[row_rank][col_rank].contiguous()

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

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:
                bias_list = module.bias.chunk(tesseract_dim * fusion_degree, dim=0)

                if fusion_degree > 1:
                    bias_list = self._deconstruct_combined_qkv(
                        bias_list,
                        tesseract_dim,
                        fusion_degree,
                        is_bias=True,
                    )

                module.bias.data = bias_list[row_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
                    module.weight.oslo_parallel[ParallelMode.TENSOR_2P5D_DEP] = dep_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                        ParallelMode.TENSOR_2P5D_COL: col_rank,
                        ParallelMode.TENSOR_2P5D_DEP: dep_rank,
                    }

        _update_module_arguments(
            module=module,
            in_features=module.weight.size()[1],
            out_features=module.weight.size()[0],
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
        module.__class__ = Linear2p5D
        return module

    def _slice_layernorm(self, module):
        row_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        col_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        dep_rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
        tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )

        data_parallel_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        pipeline_parallel_rank = self.parallel_context.get_local_rank(
            ParallelMode.PIPELINE
        )
        tensor_parallel_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        pipeline_parallel_size = self.parallel_context.get_world_size(
            ParallelMode.PIPELINE
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                weight_list = module.weight.data.chunk(tesseract_dim, dim=0)
                module.weight.data = weight_list[row_rank].contiguous()

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

        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.dim() >= 1:
                bias_list = module.bias.chunk(tesseract_dim, dim=0)
                module.bias.data = bias_list[row_rank].contiguous()

                if hasattr(module.bias, "oslo_parallel"):
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_ROW] = row_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_COL] = col_rank
                    module.bias.oslo_parallel[ParallelMode.TENSOR_2P5D_DEP] = dep_rank
                else:
                    module.bias.oslo_parallel = {
                        ParallelMode.TENSOR_2P5D_ROW: row_rank,
                        ParallelMode.TENSOR_2P5D_COL: col_rank,
                        ParallelMode.TENSOR_2P5D_DEP: dep_rank,
                    }

        _update_module_arguments(
            module=module,
            normalized_shape=module.weight.size()[0] * tesseract_dim,
            partitioned_dim=module.weight.size()[0],
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
        module.__class__ = LayerNorm2p5D
        return module

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
            row_rank = self.parallel_context.get_local_rank(
                ParallelMode.TENSOR_2P5D_ROW
            )
            col_rank = self.parallel_context.get_local_rank(
                ParallelMode.TENSOR_2P5D_COL
            )
            dep_rank = self.parallel_context.get_local_rank(
                ParallelMode.TENSOR_2P5D_DEP
            )
            tesseract_dim = self.parallel_context.get_world_size(
                ParallelMode.TENSOR_2P5D_COL
            )

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
            _update_module_arguments(
                module=module,
                in_features=module.weight.size()[1],
                out_features=module.weight.size()[0],
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
                fusion_degree=1,
                skip_bias_add=False,
                gather_output=not is_oslo_model(self.module),
                orig_module=copy.deepcopy(module.__class__),
            )
        module.__class__ = Linear2p5D

    @torch.no_grad()
    def deparallelize(self):
        self._deparallelize_linear()
        self._deparallelize_embedding()
        self._deparallelize_layernorm()
        self._rollback_mp_arguments()

    def _rollback_mp_arguments(self):
        for module in self.module.modules():
            for elem in self.tensor_parallel_mapping.update_attrs(self.module):
                if hasattr(module, elem.name):
                    tesseract_dim = self.parallel_context.get_world_size(
                        ParallelMode.TENSOR_2P5D_COL
                    )
                    expanded_arg = getattr(module, elem.name) * tesseract_dim
                    setattr(module, elem.name, expanded_arg)

    def _deparallelize_embedding(self):
        for param_name, module in self.module.named_modules():
            if module.__class__ == VocabParallelEmbedding2p5D:
                self._gather_embedding(module)

    def _deparallelize_linear(self):
        for param_name, module in self.module.named_modules():
            if module.__class__ == Linear2p5D:
                self._gather_linear(module)

    def _deparallelize_layernorm(self):
        for param_name, module in self.module.named_modules():
            if module.__class__ == LayerNorm2p5D:
                self._gather_layernorm(module)

    def _gather_embedding(self, module):
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
        if hasattr(module, "vocab_start_index") and hasattr(module, "vocab_end_index"):
            w = gather_2d(self.parallel_context, module.weight.data, tesseract_dim, col_first=True)

            assert hasattr(
                self.module, "orig_vocab_size"
            ), "wrapper's vocab embedding module must have attribute 'orig_vocab_size'."
            orig_vocab_size = self.module.orig_vocab_size

            module.weight.data = w[:, :orig_vocab_size]

            _update_module_arguments(
                module=module,
                vocab_start_index=None,
                vocab_end_index=None,
                parallel_context=None,
                num_embeddings=module.weight.size()[0],
                embedding_dim=module.weight.size()[1],
                orig_module=None
            )
        else:
            w = gather_1d(self.parallel_context, module.weight, tesseract_dim, 1)
            w = gather_1d(self.parallel_context, w, tesseract_dim, 1)
            module.weight.data = w

            _update_module_arguments(
                module=module,
                parallel_context=None,
                embedding_dim = module.weight.size()[1]
            )
        module.__class__ = nn.Embedding

    def _gather_linear(self, module: Linear2p5D):
        is_reversed = module.reversed
        fusion_degree = module.fusion_degree
        # slice_bias = module.slice_bias

        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

        w = gather_2d(self.parallel_context, module.weight.data, tesseract_dim=tesseract_dim, col_first=True)
        # print(f"w shape: {w.shape}\nweight shape: {module.weight.data.shape}")
        if fusion_degree > 1:
            w = self._reconstruct_combined_qkv(w, tesseract_dim, fusion_degree, False)
        if is_reversed:
            w = module.weight.data.t()
        module.weight.data = w

        if hasattr(module, "bias") and module.bias is not None:
            # if slice_bias is True and module.bias.dim() >= 1:
            b = gather_1d(self.parallel_context, module.bias.data, tesseract_dim, 0)
            if fusion_degree > 1:
                b = self._reconstruct_combined_qkv(b, tesseract_dim, fusion_degree, True)
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
        del module.row_rank
        del module.col_rank
        del module.dep_rank
        del module.tesseract_dim
        del module.data_parallel_rank
        del module.pipeline_parallel_rank
        del module.tensor_parallel_size
        del module.pipeline_parallel_size
        del module.reversed
        del module.fusion_degree
        del module.orig_module
        del module.gather_output
        del module.parallel_context

        module.__class__ = nn.Linear

    def _gather_layernorm(self, module):
        tesseract_dim = self.parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                w = gather_1d(self.parallel_context, module.weight.data, tesseract_dim, 0)
                module.weight.data = w

            if hasattr(module.weight, "oslo_parallel"):
                del module.weight.oslo_parallel

        if hasattr(module, "bias") and module.bias is not None:
            if module.bias.dim() >= 1:
                b = gather_1d(self.parallel_context, module.bias.data, tesseract_dim, 0)
                module.bias.data = b

            if hasattr(module.bias, "oslo_parallel"):
                del module.bias.oslo_parallel

        del module.partitioned_dim
        del module.row_rank
        del module.col_rank
        del module.dep_rank
        del module.tesseract_dim
        del module.data_parallel_rank
        del module.pipeline_parallel_rank
        del module.tensor_parallel_size
        del module.pipeline_parallel_size
        del module.orig_module
        _update_module_arguments(
            module,
            normalized_shape=module.weight.size()[0],
        )
        module.__class__ = nn.LayerNorm

    @staticmethod
    def _reconstruct_combined_qkv(tensor, tesseract_dim, fusion_degree, is_bias=False):
        last_dim = tensor.size()[-1]
        if is_bias is False:
            reshaped_w = tensor.view(tesseract_dim*fusion_degree, -1, last_dim)
            # print(f"tensor.size: {tensor.size()}, reshaped_w.size: {reshaped_w.size()}")
            recon_w = torch.cat([
                reshaped_w[:fusion_degree], reshaped_w[fusion_degree:]], 1).view(-1, last_dim).contiguous()
        else:
            reshaped_w = tensor.view(fusion_degree*tesseract_dim, -1)
            recon_w = torch.cat([
                reshaped_w[:fusion_degree], reshaped_w[fusion_degree:]], 1).view(-1, last_dim).contiguous()
        return recon_w

    @staticmethod
    def _reconstrunct_combined_qkv_bias(tensor, tessearct_dim, fusion_degree):
        tensor = [
            [tensor[j * tessearct_dim + k] for k in range(tessearct_dim)]
            for j in range(fusion_degree)
        ]
        tensor = list(map(lambda x: torch.cat([*x], dim=0), zip(*tensor)))
        tensor = [tensor[j] for j in range(tessearct_dim)]
        return tensor

    # @torch.no_grad()
    # def save_parallelized(
    #         self,
    #         new_module,
    #         save_directory: Union[str, os.PathLike],
    #         save_config: bool = True,
    #         state_dict: Optional[dict] = None,
    #         save_function: Callable = torch.save,
    #         merge_checkpoints: bool = False,
    #         mapping: Optional[dict] = None,
    #         **kwargs,
    # ):
    #     logger = getLogger("Tensor2p5D")
    #     PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"
    #
    #     if (
    #             self.parallel_context.get_world_size(ParallelMode.TENSOR) == 1
    #             and self.parallel_context.get_world_size(ParallelMode.PIPELINE) == 1
    #     ):
    #         if dist.get_rank() == 0:
    #             self.save_pretrained(
    #                 save_directory=save_directory,
    #                 save_config=save_config,
    #                 state_dict=state_dict,
    #                 save_function=save_function,
    #                 **kwargs,
    #             )
    #         dist.barrier()
    #         return None
    #
    #     if merge_checkpoints:
    #         model_to_save = self.__class__(
    #             module=new_module,
    #             parallel_context=self.parallel_context,
    #             mapping=mapping,
    #             module_args=self.config
    #         ).eval()
    #
    #         if state_dict is None:
    #             state_dict = self.state_dict()
    #
    #         model_to_save.load_state_dict(state_dict)
    #         allocate_params(model_to_save, self.parallel_context)
    #
    #         if self.parallel_context.get_world_size(ParallelMode.TENSOR) > 1:
    #             model_to_save.deparallelize()
    #
    #         if dist.get_rank() == 0:
    #             if is_huggingface_model(model_to_save.module):
    #                 model_to_save.module.save_pretrained(
    #                     save_directory=save_directory,
    #                     save_config=save_config,
    #                     save_function=save_function,
    #                     **kwargs,
    #                 )
    #             else:
    #                 if save_config:
    #                     with open(os.path.join(save_directory, "config.json"), "w") as f:
    #                         json.dump(self.config, f)
    #                 save_function(
    #                     model_to_save,
    #                     os.path.join(save_directory, "pytorch_model.bin"),
    #                 )
    #         del model_to_save
    #
    #         dist.barrier()
    #         return None
    #
    #     if os.path.isfile(save_directory):
    #         logger.error(
    #             f"Provided path ({save_directory}) should be a directory, not a file"
    #         )
    #         return
    #
    #     os.makedirs(save_directory, exist_ok=True)
    #
    #     # Only save the model itself if we are using distributed training
    #     model_to_save = unwrap_parallel(self)
    #
    #     # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
    #     # we currently don't use this setting automatically, but may start to use with v5
    #     dtype = get_parameter_dtype(model_to_save)
    #     model_to_save.config.torch_dtype = str(dtype).split(".")[1]
    #
    #     # Attach architecture to the config
    #     model_to_save.config.architectures = [model_to_save.__class__.__name__]
    #
    #     # Save the config
    #     if save_config:
    #         model_to_save.config.save_pretrained(save_directory)
    #
    #     # Save the model
    #     if state_dict is None:
    #         state_dict = model_to_save.state_dict()
    #
    #     # Handle the case where some state_dict keys shouldn't be saved
    #     if getattr(self, "_keys_to_ignore_on_save", None) is not None:
    #         state_dict = {
    #             k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save
    #         }
    #
    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     weights_name = PARALLELIZED_WEIGHTS_NAME
    #     weights_name = weights_name.replace(
    #         "tp_0", f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}"
    #     )
    #     weights_name = weights_name.replace(
    #         "pp_0", f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}"
    #     )
    #
    #     output_model_file = os.path.join(save_directory, weights_name)
    #
    #     if self.parallel_context.get_world_size(ParallelMode.DATA) > 1:
    #         if self.parallel_context.get_local_rank(ParallelMode.DATA) == 0:
    #             save_function(state_dict, output_model_file)
    #     else:
    #         save_function(state_dict, output_model_file)
    #
    #     dist.barrier()
    #     logger.info(f"Model weights saved in {output_model_file}")
    #
    # def from_parallelized(self, path):
    #     """
    #     Example:
    #     >>> model = AnyModel()
    #     >>> model = TensorParallel(model, ...)
    #     >>> model.from_parallelized(path)
    #     """
    #     PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"
    #     parallelized_model_path = path
    #
    #     file_names = {
    #         os.path.join(
    #             parallelized_model_path,
    #             PARALLELIZED_WEIGHTS_NAME.replace("tp_0", f"tp_{tp}").replace(
    #                 "pp_0", f"pp_{pp}"
    #             ),
    #         )
    #         for tp in range(self.parallel_context.get_world_size(ParallelMode.TENSOR))
    #         for pp in range(self.parallel_context.get_world_size(ParallelMode.PIPELINE))
    #     }
    #
    #     if os.path.isdir(parallelized_model_path):
    #         if all(os.path.isfile(file_name) for file_name in file_names):
    #             state_dict = torch.load(
    #                 os.path.join(
    #                     parallelized_model_path,
    #                     PARALLELIZED_WEIGHTS_NAME.replace(
    #                         "tp_0",
    #                         f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}",
    #                     ).replace(
    #                         "pp_0",
    #                         f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}",
    #                     ),
    #                 )
    #             )
    #
    #             if getattr(self, "_keys_to_ignore_on_save", None) is not None:
    #                 state_dict = {
    #                     k: v
    #                     for k, v in state_dict.items()
    #                     if k not in self._keys_to_ignore_on_save
    #                 }
    #
    #             self.load_state_dict(state_dict=state_dict, strict=False)
    #
    #         else:
    #             raise FileNotFoundError(
    #                 f"all the {file_names} are necessary. "
    #                 f"but some of them do not exist. Please check your checkpoint files."
    #             )
    #     else:
    #         raise NotADirectoryError(
    #             f"directory named {parallelized_model_path} is not valid. "
    #         )
