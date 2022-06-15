from typing import Union, Optional, Callable

import os
import json
from operator import xor

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoConfig

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
from oslo.torch.nn.parallel.tensor_parallel.mapping import (
    TensorParallelMapping,
)
from oslo.transformers.mapping_utils import (
    _TensorParallelMappingForHuggingFace,
)
from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    unwrap_parallel,
    get_parallel_context,
    is_huggingface_model,
    allocate_params,
    get_parameter_dtype
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
        mapping: dict = None,
        module_args: dict = None
    ):
        super().__init__()
        if is_huggingface_model(module):
            assert module_args is None, "module_args must not be provided in huggingface module."
        else:
            assert isinstance(module_args, dict), "module_args must be a dict."

        self.parallel_context = get_parallel_context(module, parallel_context)
        module = self._resize_vocab_size(module, self.parallel_context)
        module = self._resize_num_classes(module, self.parallel_context, mapping)

        if self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_1D:
            self.module = _TensorParallel1D(module, self.parallel_context, mapping, module_args)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2D:
            self.module = _TensorParallel2D(module, self.parallel_context, mapping, module_args)
        elif self.parallel_context.tensor_parallel_mode == ParallelMode.TENSOR_2P5D:
            self.module = _TensorParallel2p5D(module, self.parallel_context, mapping, module_args)
        else:
            raise ValueError(
                "currently, only 1d, 2d, 2p5d tensor parallelism is supported."
            )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @staticmethod
    def _resize_vocab_size(model, parallel_context):
        unwrapped_model = unwrap_parallel(model)

        assert hasattr(
            unwrapped_model, "get_input_embeddings"
        ), "model object must have `get_input_embeddings` method."

        module = unwrapped_model.get_input_embeddings()

        vocab_size, embedding_dim = module.weight.size()
        new_vocab_size = vocab_size

        world_size = parallel_context.get_world_size(ParallelMode.TENSOR)
        while new_vocab_size % world_size != 0:
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
        world_size = parallel_context.get_world_size(ParallelMode.TENSOR)

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

                    while new_out_features % world_size != 0:
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

    @torch.no_grad()
    def save_parallelized(
            self,
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            merge_checkpoints: bool = False,
            mapping: Optional[dict] = None,
            **kwargs,
    ):
        unwrapped_model = unwrap_parallel(self.module.module)
        if is_huggingface_model(unwrapped_model):
            new_module = unwrapped_model.__class__(self.module.config)
        else:
            new_module = unwrapped_model.__class__(**self.module.config)
        new_module = self._resize_vocab_size(new_module, self.parallel_context)
        new_module = self._resize_num_classes(new_module, self.parallel_context, mapping)
        return self.module.save_parallelized(
            new_module,
            save_directory,
            save_config,
            state_dict,
            save_function,
            merge_checkpoints,
            mapping,
            **kwargs,
        )

    @staticmethod
    def get_module_args(module):
        state_dict = module.state_dict()
        return {
            key: value.shape for key, value in state_dict.items()
        }

    def from_parallelized(self, path):
        return self.module.from_parallelized(path)

    # @classmethod
    # def from_parallelized(cls, parallelized_model_path, parallel_context,
    #                       huggingface_task_class=None, base_model_class=None):
    #     """
    #     :param parallelized_model_path:  path to the parallelized model
    #     :param parallel_context: parallel context
    #     :param huggingface_task_class: huggingface task class ex. BertForSequenceClassification
    #     :param base_model_class: custom model's class field ex. CustomModel
    #     :return: TensorParallelWrapper
    #
    #     Examples:
    #
    #     >>> # huggingface model
    #     >>> huggingface_parallelized = TensorParallel.from_parallelized(
    #     >>>   "bert-base-uncased",
    #     >>>    parallel_context,
    #     >>>    huggingface_task_class=BertForSequenceClassification,
    #     >>> )
    #
    #     >>> # custom model
    #     >>> from path.to.custom.model import CustomModel
    #     >>> custom_parallelized = TensorParallel.from_parallelized(
    #     >>>     "/path/to/custom_model",
    #     >>>    parallel_context,
    #     >>>    base_model_class=CustomModel,
    #     >>>)
    #     """
    #     assert xor(huggingface_task_class is None, base_model_class is None), \
    #         "`huggingface_task_class` and `orig_model` must be input only one of them."
    #
    #     if base_model_class is None:
    #         config = AutoConfig.from_pretrained(parallelized_model_path)
    #         base_model = huggingface_task_class(config)
    #         config = None
    #     else:
    #         config_path = os.path.join(parallelized_model_path, "config.json")
    #         if os.path.exists(config_path):
    #             with open(config_path, "r") as f:
    #                 config = dict(json.load(f))
    #         else:
    #             raise ValueError(f"`config.json` is not found in {parallelized_model_path}.")
    #         base_model = base_model_class(**config)
    #
    #     self = cls(base_model, parallel_context, None, config)
    #     allocate_params(self, parallel_context)
    #     self.module.from_parallelized(parallelized_model_path)
    #
    #     return self
