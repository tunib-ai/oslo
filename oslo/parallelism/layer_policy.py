# Copyright 2021 TUNiB Inc.

from abc import ABC
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Layer:
    """Data class to describe a layer in the model"""

    module: nn.Module = None
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    replace: dict = None
    n_fused: int = None
    reversed: bool = None
    parallel: bool = True
    input_is_parallel: bool = True
    gather_output: bool = False
    tied_embedding: nn.Module = None


class LayerPolicy(ABC):
    """
    Layer policy for model parallelism and kernel fusion
    You can check more details here: https://github.com/tunib-ai/parallelformers/blob/main/POLICY.md

    References:
        The design of the LayerPolicy class is inspired by Microsoft DeepSpeed.
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py
    """

    @staticmethod
    def reduce_arguments(layer, world_size, config):
        ...

    @staticmethod
    def fused_modules():
        return {}

    @staticmethod
    def attn_qkv(layer, config):
        return []

    @staticmethod
    def attn_out(layer, config):
        return []

    @staticmethod
    def attn_norm(layer, config):
        return []

    @staticmethod
    def mlp_in(layer, config):
        return []

    @staticmethod
    def mlp_out(layer, config):
        return []

    @staticmethod
    def mlp_norm(layer, config):
        return []

    @staticmethod
    def word_embedding(model, config):
        return []

    @staticmethod
    def preblock_layers(model, config):
        return []

    @staticmethod
    def block_layers(model, config):
        return []

    @staticmethod
    def postblock_layers(model, config):
        return []

    @staticmethod
    def copy_to_all(layer, config):
        return []

    @staticmethod
    def original_layer_class():
        raise NotImplementedError
