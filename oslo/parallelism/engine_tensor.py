# Copyright 2021 TUNiB Inc.

from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from oslo.parallelism.mpu import MPU, Layer, LayerPolicy


class TensorParallelEngine(object):
    """
    Tensor model parallelism engine designed by TUNiB.

    Notes:
        This engine is based on Tensor replacement mechanism proposed by Parallelformers.
        You can easily parallelize your model by providing a simple policy object.
        Note it should be performed later than pipeline parallelism.

    Args:
        mpu (MPU): model parallel unit
        policy (LayerPolicy): layer policy
        head_layers (Optional[List[Layer]]): head layer policies

    References:
        Parallelformers: https://github.com/tunib-ai/parallelformers
    """

    def __init__(
        self, mpu: MPU, policy: LayerPolicy, head_layers: Optional[List[Layer]] = None
    ):
        self.mpu = mpu
        self.policy = policy
        self.device = torch.cuda.current_device()
        self.head_layers = head_layers if head_layers is not None else []

        # for 3d parallelism
        self.is_pipeline_parallelized = mpu.get_pipeline_parallel_world_size() > 1
        self.block_start = 0
        self.block_stop = float("inf")

    def parallelize(self, model: nn.Module):
        """
        Parallelize tensor dimension

        Args:
            model (nn.Module): model object
        """

        config = model.config
        model = model.base_model

        if self.is_pipeline_parallelized:
            block_layers = self.policy.block_layers(model, model.config)
            num_layers = len(block_layers)

            # compute range of partition for current rank
            partitions = self.mpu.make_pipeline_partition(
                num_items=num_layers,
                num_parts=self.mpu.get_pipeline_parallel_world_size(),
            )

            self.block_start = partitions[self.mpu.get_pipeline_parallel_rank()]
            self.block_stop = partitions[self.mpu.get_pipeline_parallel_rank() + 1]

        if dist.is_initialized() and self.mpu is not None:
            gpu_index = self.mpu.get_tensor_parallel_rank()
            world_size = self.mpu.get_tensor_parallel_world_size()
        else:
            gpu_index = 0
            world_size = 1

        self._parallelize_preblock_layers(
            model=model,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        self._parallelize_block_layers(
            model=model,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        self._parallelize_postblock_layers(
            model=model,
            head_layers=self.head_layers,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        self._parallelize_word_embedding(
            model=model,
            gpu_index=gpu_index,
            world_size=world_size,
            config=config,
        )

        if self.mpu.get_pipeline_parallel_world_size() < 2:
            non_cuda_tensors = []
            for k, v in dict(model.state_dict()).items():
                if not v.is_cuda:
                    if torch.is_tensor(v):
                        non_cuda_tensors.append(k)

            if len(non_cuda_tensors) > 0:
                raise ValueError(f"{non_cuda_tensors} are not CUDA tensors now.")

    def _parallelize_word_embedding(
        self,
        model: nn.Module,
        gpu_index: int,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelize word embedding layer

        Args:
            model (nn.Module): model object
            gpu_index (int): partition index of tensor
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """
        for layer in self.policy.word_embedding(model, config):
            weight = layer.weight

            num_embeddings = weight.size(0)
            chunked_weight = torch.chunk(weight, world_size, dim=0)
            layer.weight.data = chunked_weight[gpu_index].to(self.device)

            attributes = {"mpu": self.mpu, "num_embeddings": num_embeddings}
            self._postprocess(layer, attributes)

    def _parallelize_preblock_layers(
        self,
        model: nn.Module,
        gpu_index: int,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelized pre-block layers
        (e.g. positional embedding, ...)

        Args:
            model (nn.Module): model object
            gpu_index (int): partition index of tensor
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """

        preblock_layers = self.policy.preblock_layers(
            model=model,
            config=config,
        )

        preblock_layers = self._column_slice(
            preblock_layers,
            world_size=world_size,
            gpu_index=gpu_index,
            to_gpu=True,
        )

        for param in preblock_layers:
            sizes = self._get_layer_size(param)
            attributes = {
                "mpu": self.mpu,
                "in_features": sizes[0],
                "out_features": sizes[1],
                "reversed": param.reversed,
                "input_is_parallel": param.input_is_parallel,
                "gather_output": param.gather_output,
            }
            self._postprocess(param, attributes)

    def _parallelize_postblock_layers(
        self,
        model: nn.Module,
        head_layers: List[Layer],
        gpu_index: int,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelize post-block layers
        (e.g. final layernorm, head layers, ...)

        Args:
            model (nn.Module): model object
            head_layers (List[Layer]): list of head layer policies
            gpu_index (int): partition index of tensor
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """

        # head layers + post-block layers
        postblock_layers = (
            self.policy.postblock_layers(model=model, config=config) + head_layers
        )

        postblock_layers = self._column_slice(
            postblock_layers,
            world_size=world_size,
            gpu_index=gpu_index,
            to_gpu=True,
        )

        for param in postblock_layers:
            sizes = self._get_layer_size(param)
            attributes = {
                "mpu": self.mpu,
                "in_features": sizes[0],
                "out_features": sizes[1],
                "reversed": param.reversed,
                "input_is_parallel": param.input_is_parallel,
                "gather_output": param.gather_output,
            }
            self._postprocess(param, attributes)

    def _parallelize_block_layers(
        self,
        model: nn.Module,
        gpu_index: int,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelize block layers

        Args:
            model (nn.Module): model object
            gpu_index (int): partition index of tensor
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """
        for i, layer in enumerate(self.policy.block_layers(model, config)):
            # Parallelize only original layer class
            if not isinstance(layer, self.policy.original_layer_class()):
                continue

            # If the model is pipelined parallelized,
            # we need to send parameters to the gpu at the correct stage.
            to_gpu = self.block_start <= i < self.block_stop

            if hasattr(config, "num_attention_heads"):
                assert (
                    config.num_attention_heads >= world_size
                ), "number of attention heads must be bigger than tensor parallel size."
            if hasattr(config, "hidden_size"):
                assert (
                    config.hidden_size >= world_size
                ), "hidden size must be bigger than tensor parallel size."

            self.policy.reduce_arguments(
                layer=layer,
                world_size=world_size,
                config=config,
            )

            parameters = (
                self._column_slice(
                    self.policy.attn_qkv(layer, config),
                    gpu_index=gpu_index,
                    world_size=world_size,
                    to_gpu=to_gpu,
                )
                + self._row_slice(
                    self.policy.attn_out(layer, config),
                    gpu_index=gpu_index,
                    world_size=world_size,
                    to_gpu=to_gpu,
                )
                + self._column_slice(
                    self.policy.mlp_in(layer, config),
                    gpu_index=gpu_index,
                    world_size=world_size,
                    to_gpu=to_gpu,
                )
                + self._row_slice(
                    self.policy.mlp_out(layer, config),
                    gpu_index=gpu_index,
                    world_size=world_size,
                    to_gpu=to_gpu,
                )
            )

            for param in (
                self.policy.attn_norm(layer, config)
                + self.policy.mlp_norm(layer, config)
                + self.policy.copy_to_all(layer, config)
            ):
                if to_gpu is True:
                    if param.weight is not None:
                        param.weight.data = param.weight.to(self.device)
                    if param.bias is not None:
                        param.bias.data = param.bias.to(self.device)

                parameters.append(param)

            for param in parameters:
                sizes = self._get_layer_size(param)
                attributes = {
                    "mpu": self.mpu,
                    "in_features": sizes[0],
                    "out_features": sizes[1],
                    "reversed": param.reversed,
                    "input_is_parallel": param.input_is_parallel,
                    "gather_output": param.gather_output,
                }
                self._postprocess(param, attributes)

    def _slice_layer(
        self,
        layer: Layer,
        dim: int,
        gpu_index: int,
        world_size: int,
        slice_bias: bool,
        to_gpu: bool,
    ) -> Layer:
        """
        Slice linear layer as described Megatron-LM paper

        Args:
            layer (Layer): layer object
            dim (int): dimension to slice
            gpu_index (int): partition index of tensor
            world_size (int): tensor model parallel world size
            slice_bias (bool): slice bias vector or not
            to_gpu (bool): move to gpu or not

        Returns:
            Layer: layer object that contains sliced parameters
        """

        dim = dim if not layer.reversed else abs(dim - 1)
        n_fused = 1 if not layer.n_fused else layer.n_fused

        if layer.weight is not None:
            if layer.parallel and layer.weight.dim() >= 1:
                weight = layer.weight.chunk(n_fused * world_size, dim=dim)
                if n_fused > 1:
                    weight = self._realign_fused_tensors(weight, world_size)
                layer.weight.data = weight[gpu_index]

            if to_gpu is True:
                layer.weight.data = layer.weight.to(self.device)

        if layer.bias is not None:
            if slice_bias is True:
                if layer.parallel and layer.bias.dim() >= 1:
                    bias = layer.bias.chunk(n_fused * world_size, dim=0)
                    if n_fused > 1:
                        bias = self._realign_fused_tensors(bias, world_size)
                    layer.bias.data = bias[gpu_index]

            if to_gpu is True:
                layer.bias.data = layer.bias.to(self.device)

        return layer

    def _column_slice(
        self,
        layers: List[Layer],
        gpu_index: int,
        world_size: int,
        to_gpu: bool,
    ) -> List[Layer]:
        """
        Slice tensor by column dimension

        Args:
            layers (List[Layer]): list of layer object
            gpu_index (int): partition index of tensor
            world_size (int): world size
            to_gpu (bool): move to gpu or not

        Returns:
            List[Layer]: list of layer object that contains sliced parameters
        """

        return [
            self._slice_layer(
                layer=layer,
                dim=0,
                gpu_index=gpu_index,
                world_size=world_size,
                slice_bias=True,
                to_gpu=to_gpu,
            )
            for layer in layers
        ]

    def _row_slice(
        self,
        layers: List[Layer],
        gpu_index: int,
        world_size: int,
        to_gpu: bool,
    ) -> List[Layer]:
        """
        Slice tensor by row dimension

        Args:
            layers (List[Layer]): list of layer object
            gpu_index (int): partition index of tensor
            world_size (int): world size
            to_gpu (bool): move to gpu or not

        Returns:
            List[Layer]: list of layer object that contains sliced parameters
        """

        return [
            self._slice_layer(
                layer=layer,
                dim=1,
                gpu_index=gpu_index,
                world_size=world_size,
                slice_bias=False,
                to_gpu=to_gpu,
            )
            for layer in layers
        ]

    @staticmethod
    def _realign_fused_tensors(tensor, world_size):
        ranks = (len(tensor) + world_size - 1) // world_size
        tensor = [tensor[i * world_size : (i + 1) * world_size] for i in range(ranks)]
        tensor = list(map(lambda x: torch.cat([*x], dim=-1), zip(*tensor)))
        return tensor

    @staticmethod
    def _postprocess(layer, attributes):
        if layer.module is not None:
            for attr_key, attr_val in attributes.items():
                setattr(layer.module, attr_key, attr_val)
            if layer.replace is not None:
                for replace_module in layer.replace.values():
                    layer.module.__class__ = replace_module

    @staticmethod
    def _get_layer_size(layer):
        if layer.weight is not None:
            in_features = layer.weight.size(0)
            if layer.weight.dim() >= 2:
                out_features = layer.weight.size(1)
            else:
                out_features = "None"
        else:
            in_features = "None"
            out_features = "None"

        return in_features, out_features


class TensorDeparallelEngine(object):
    """
    Tensor model deparallelism engine designed by TUNiB.

    You can easily deparallelize your model by providing a simple policy object.
    It should be performed before pipeline deparallelism.

    Args:
        mpu (MPU): model parallel unit
        policy (LayerPolicy): layer policy
        head_layers (Optional[List[Layer]]): head layer policies
    """

    def __init__(
        self, mpu: MPU, policy: LayerPolicy, head_layers: Optional[List[Layer]] = None
    ):
        self.mpu = mpu
        self.policy = policy
        self.device = torch.cuda.current_device()
        self.head_layers = head_layers if head_layers is not None else []

    @torch.no_grad()
    def deparallelize(self, model: nn.Module):
        """
        Deparallelize tensor dimension

        Args:
            model (nn.Module): model object
        """

        config = model.config
        model = model.base_model

        if dist.is_initialized() or self.mpu is not None:
            world_size = self.mpu.get_tensor_parallel_world_size()
        else:
            world_size = 1

        self._deparallelize_preblock_layers(
            model=model,
            world_size=world_size,
            config=config,
        )

        self._deparallelize_block_layers(
            model=model,
            world_size=world_size,
            config=config,
        )

        self._deparallelize_postblock_layers(
            model=model,
            world_size=world_size,
            config=config,
            head_layers=self.head_layers,
        )

        self._deparallelize_word_embedding(
            model=model,
            config=config,
        )

        if self.mpu.get_pipeline_parallel_world_size() < 99:
            cuda_tensors = []
            for k, v in dict(model.state_dict()).items():
                if v.is_cuda:
                    if torch.is_tensor(v):
                        cuda_tensors.append(k)

            if len(cuda_tensors) > 0:
                raise ValueError(f"{cuda_tensors} are CUDA tensors now.")

    @torch.no_grad()
    def _deparallelize_word_embedding(self, model: nn.Module, config: PretrainedConfig):
        """
        Deparallelize word embedding layer

        Args:
            model (nn.Module): model object
            config (PretrainedConfig): config object
        """
        for layer in self.policy.word_embedding(model, config):
            weight = layer.weight

            if not layer.weight.is_cuda:
                weight = weight.cuda()

            gathered_weight = self.mpu._gather(weight, dim=0)
            num_embeddings = gathered_weight.size(0)
            layer.weight.data = gathered_weight.cpu()
            self._postprocess(layer, {"num_embeddings": num_embeddings})

    @torch.no_grad()
    def _deparallelize_preblock_layers(
        self,
        model: nn.Module,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelized pre-block layers
        (e.g. positional embedding, ...)

        Args:
            model (nn.Module): model object
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """
        preblock_layers = self.policy.preblock_layers(
            model=model,
            config=config,
        )

        preblock_layers = self._column_merge(
            preblock_layers,
            world_size=world_size,
        )

        for param in preblock_layers:
            sizes = self._get_layer_size(param)
            attributes = {
                "in_features": sizes[0],
                "out_features": sizes[1],
                "nx": sizes[0],
                "nf": sizes[1],
            }

            self._postprocess(param, attributes)

    @torch.no_grad()
    def _deparallelize_postblock_layers(
        self,
        model: nn.Module,
        head_layers: List[LayerPolicy],
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelize post-block layers
        (e.g. final layernorm, head layers, ...)

        Args:
            model (nn.Module): model object
            head_layers (List[LayerPolicy]): list of head layer policies
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """
        postblock_layers = (
            self.policy.postblock_layers(model=model, config=config) + head_layers
        )

        postblock_layers = self._column_merge(
            postblock_layers,
            world_size=world_size,
        )

        for param in postblock_layers:
            sizes = self._get_layer_size(param)
            attributes = {
                "in_features": sizes[0],
                "out_features": sizes[1],
                "nx": sizes[0],
                "nf": sizes[1],
            }

            self._postprocess(param, attributes)

    def _deparallelize_block_layers(
        self,
        model: nn.Module,
        world_size: int,
        config: PretrainedConfig,
    ):
        """
        Parallelize block layers

        Args:
            model (nn.Module): model object
            world_size (int): tensor model parallel world size
            config (PretrainedConfig): config object
        """

        for layer in self.policy.block_layers(model, config):
            # deparallelize only original layer class
            if not isinstance(layer, self.policy.original_layer_class()):
                continue

            parameters = (
                self._column_merge(
                    self.policy.attn_qkv(layer, config),
                    world_size=world_size,
                )
                + self._row_merge(
                    self.policy.attn_out(layer, config),
                    world_size=world_size,
                )
                + self._column_merge(
                    self.policy.mlp_in(layer, config),
                    world_size=world_size,
                )
                + self._row_merge(
                    self.policy.mlp_out(layer, config),
                    world_size=world_size,
                )
            )

            for param in (
                self.policy.attn_norm(layer, config)
                + self.policy.mlp_norm(layer, config)
                + self.policy.copy_to_all(layer, config)
            ):
                if param.weight is not None:
                    param.weight.data = param.weight.cpu()
                if param.bias is not None:
                    param.bias.data = param.bias.cpu()

                parameters.append(param)

            for param in parameters:
                sizes = self._get_layer_size(param)
                attributes = {
                    "in_features": sizes[0],
                    "out_features": sizes[1],
                    "nx": sizes[0],
                    "nf": sizes[1],
                }

                self._postprocess(param, attributes)

    @torch.no_grad()
    def _realign_fused_tensors(self, tensors, n_fused, dim):
        result_tensors = {i: [] for i in range(n_fused)}
        for tensor in tensors:
            chunks = tensor.chunk(n_fused, dim=dim)
            for i, chunk in enumerate(chunks):
                result_tensors[i].append(chunk)
        for key, val in result_tensors.items():
            result_tensors[key] = torch.cat(val, dim=dim)
        return torch.cat(list(result_tensors.values()), dim=dim)

    @torch.no_grad()
    def _merge_layer(
        self,
        layer: Layer,
        dim: int,
        world_size: int,
        merge_bias: bool,
    ) -> Layer:
        """
        Merge tensors by rows or columns as described in the Megatron-LM paper
        """

        dim = dim if not layer.reversed else abs(dim - 1)
        n_fused = 1 if not layer.n_fused else layer.n_fused

        if layer.weight is not None:
            if layer.parallel and layer.weight.dim() >= 1:
                if not layer.weight.is_contiguous():
                    layer.weight.data = layer.weight.contiguous()
                if not layer.weight.is_cuda:
                    layer.weight.data = layer.weight.to(self.device)

                tensor_list = [
                    torch.zeros_like(layer.weight) for _ in range(world_size)
                ]
                dist.all_gather(
                    tensor_list,
                    layer.weight,
                    group=self.mpu.get_tensor_parallel_group(),
                )
                if n_fused > 1:
                    output = self._realign_fused_tensors(tensor_list, n_fused, dim=dim)
                else:
                    output = torch.cat(tensor_list, dim=dim)

                layer.weight.data = output

            # move every weight tensors to cpu
            layer.weight.data = layer.weight.cpu()

        if layer.bias is not None:
            if merge_bias is True:
                if layer.parallel and layer.bias.dim() >= 1:
                    if not layer.bias.is_contiguous():
                        layer.bias.data = layer.bias.contiguous()
                    if not layer.bias.is_cuda:
                        layer.bias.data = layer.bias.to(self.device)

                    tensor_list = [
                        torch.zeros_like(layer.bias) for _ in range(world_size)
                    ]
                    dist.all_gather(
                        tensor_list,
                        layer.bias,
                        group=self.mpu.get_tensor_parallel_group(),
                    )
                    if n_fused > 1:
                        output = self._realign_fused_tensors(
                            tensor_list, n_fused, dim=0
                        )
                    else:
                        output = torch.cat(tensor_list, dim=0)

                    layer.bias.data = output

            # move every bias tensors to cpu
            layer.bias.data = layer.bias.cpu()

        return layer

    @torch.no_grad()
    def _column_merge(self, layers, world_size):
        return [
            self._merge_layer(
                layer=layer,
                dim=0,
                world_size=world_size,
                merge_bias=True,
            )
            for layer in layers
        ]

    @torch.no_grad()
    def _row_merge(self, layers, world_size):
        return [
            self._merge_layer(
                layer=layer,
                dim=1,
                world_size=world_size,
                merge_bias=False,
            )
            for layer in layers
        ]

    @staticmethod
    @torch.no_grad()
    def _postprocess(layer, attributes):
        if layer.module is not None:
            for attr_key, attr_val in attributes.items():
                setattr(layer.module, attr_key, attr_val)
            if layer.replace is not None:
                for orig_module in layer.replace.keys():
                    layer.module.__class__ = orig_module

    @staticmethod
    def _get_layer_size(layer):
        if layer.weight is not None:
            in_features = layer.weight.size(0)
            if layer.weight.dim() >= 2:
                out_features = layer.weight.size(1)
            else:
                out_features = "None"
        else:
            in_features = "None"
            out_features = "None"

        return in_features, out_features
