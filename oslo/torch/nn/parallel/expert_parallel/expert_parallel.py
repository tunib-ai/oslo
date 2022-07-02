import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._seed.helper import seed

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.utils import ParallelWrapper
from oslo.torch.nn.parallel.utils import is_huggingface_model, _update_module_arguments

from oslo.transformers.mapping_utils import _ExpertParallelMappingForHuggingFace

from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext
from oslo.torch.nn.parallel.expert_parallel.mapping import ExpertParallelMapping

from oslo.torch.nn.parallel.expert_parallel.experts import Experts
from oslo.torch.nn.parallel.expert_parallel.layers import (
    ExpertParallelFrontBlockDS,
    ExpertParallelBehindBlockDS,
    TopKGate,
)

from oslo.torch.nn.parallel.expert_parallel._ops import OSLO_EP_KERNEL_FLAG

from oslo.torch.nn.parallel.expert_parallel.utils import (
    UniformNoiseSampler,
    NormalNoiseSampler,
)


class ExpertParallel(ParallelWrapper):
    """
    A class to wrap the given model for expert parallelization

    Args:
        model: model to wrap for expert paralleization
        parallel_context: global parallel context
        use_kernel_optim: flag to use kernel optimization
        num_experts: number of experts
        top_k: the number of experts for each token to be dispatched
        capacity_factor_train: capacity of each expert for training
        capacity_factor_eval: capacity of each expert for evaluation
        min_capacity: minimum capacity of each expert
        select_policy: policy to select top-1 expert ("first" or "random")
        noisy_policy: policy to generate and add noise ("Jitter" or "Gaussian")
        drop_tks: flag to drop tokens in the case that the number of dispatched tokens is larger than capacity
        use_residual: flag to use residual network proposed by
                      DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
        mapping: mapping for each module to expert-parallelize

    Notes:
        1. Similar design with `torch.nn.parallel.DistributedDataParallel`
        2. Support data parallel for non-expert paramete

    Examples:
        >>> from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel

        >>> model = AnyTransformerModel()
        >>> ep_wrapper = ExpertParallel(model, parallel_context=..., ...)
        >>> optimizer = AnyOptimizer(ep_wrapper.parameters(), lr=3e-5)

        >>> output = ep_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        parallel_context: ParallelContext,
        use_kernel_optim=True,
        num_enc_experts=None,
        num_dec_experts=None,
        top_k: int = 2,
        capacity_factor_train: float = 1.0,
        capacity_factor_eval: float = 1.0,
        min_capacity: int = 4,
        select_policy: str = "first",
        noisy_policy: str = None,
        use_rts: bool = True,
        drop_tokens: bool = True,
        use_residual: bool = None,
        mapping: object = None,
    ):
        super().__init__()

        self.model = model

        use_kernel_optim = OSLO_EP_KERNEL_FLAG and use_kernel_optim
        self.ep_context = ExpertParallelContext(parallel_context, use_kernel_optim)
        self.ep_context.setup(parallel_context.seed)
        self.ep_context.reset_loss()

        self.device = torch.cuda.current_device()

        self.use_residual = use_residual
        if use_residual is None:
            self.use_residual = True if top_k == 1 else False

        if noisy_policy is None:
            noisy_policy = "Jitter" if use_residual else "RSample"

        self.top_k = top_k
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_policy = noisy_policy
        self.use_rts = use_rts
        self.drop_tokens = drop_tokens

        if is_huggingface_model(model):
            mapping = _ExpertParallelMappingForHuggingFace().get_mapping(model)
        else:
            assert (
                mapping is not None
            ), "`mapping` must be input if the model is not huggingface model."
            mapping = mapping.get_mapping(model)

        self.expert_parallel_mapping = ExpertParallelMapping(mapping)

        self.link_info = dict()

        self.enc_layer_ids, self.dec_layer_ids = self._get_architecture_info()

        self.num_experts = dict()
        self.num_experts["enc"] = self._get_num_experts(
            num_enc_experts, self.enc_layer_ids
        )
        self.num_experts["dec"] = self._get_num_experts(
            num_dec_experts, self.dec_layer_ids
        )

        self._parallelize()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def _parallelize(self):
        self._parallelize_module()
        self.to(self.device)
        # self._sync_ep_model_param()

    def _parallelize_module(self):
        to_parallelize = [
            (module_name, module) for module_name, module in self.model.named_modules()
        ]
        for module_name, module in to_parallelize:
            if self.expert_parallel_mapping.is_front_parallel(self.model, module_name):
                self._wrap_front_ds(
                    module,
                    module_name,
                    reversed=self.expert_parallel_mapping.is_reversed_param(
                        self.model, module_name
                    ),
                )
                module.__class__ = ExpertParallelFrontBlockDS
            elif self.expert_parallel_mapping.is_behind_parallel(
                self.model, module_name
            ):
                self._wrap_behind_ds(
                    module,
                    module_name,
                    reversed=self.expert_parallel_mapping.is_reversed_param(
                        self.model, module_name
                    ),
                )
                module.__class__ = ExpertParallelBehindBlockDS

        return

    def _get_num_experts(self, num_experts, layer_ids):
        num_experts = (
            self.ep_context.get_world_size() if num_experts is None else num_experts
        )

        if len(layer_ids) == 0:
            return None

        if type(num_experts) is int:
            assert num_experts > 0, "The Number of Experts must be Positive."
            num_experts = {cur_id: num_experts for cur_id in layer_ids}
        elif type(num_experts) is dict:
            assert (
                num_experts.keys() != layer_ids
            ), "The Keys of Experts Dictionary must be equal to the Set of Layer Ids"
        else:
            raise TypeError("num_enc_experts or num_dec_experts must be int or dict")

        return num_experts

    def _get_module_role(self, module_name):
        elem = self.expert_parallel_mapping.search(self.model, module_name)
        if elem is None:
            return

        if elem.enc_name is not None and elem.enc_name in module_name:
            return "enc"

        if elem.dec_name is not None and elem.dec_name in module_name:
            return "dec"

        return

    def _get_architecture_info(self):
        enc_layer_ids, dec_layer_ids = set(), set()
        for module_name, module in self.model.named_modules():
            role = self._get_module_role(module_name)
            if role is None:
                continue

            if role == "enc":
                enc_layer_ids.add(self._extract_layer_id(module_name))
            elif role == "dec":
                dec_layer_ids.add(self._extract_layer_id(module_name))
            else:
                raise ValueError(
                    "The mapping information about Encoder/Decoder is wrong."
                )

        return enc_layer_ids, dec_layer_ids

    def _extract_layer_id(self, module_name):
        layer_info = self.expert_parallel_mapping.get_layer_info(
            self.model, module_name
        )

        spl_modules = module_name.split(".")
        spl_layer_info = layer_info.split(".")

        layer_ids = list()

        for cur_layer_info in spl_layer_info:
            to_find = spl_modules.index(cur_layer_info)
            layer_ids.append(int(spl_modules[to_find + 1]))

        return tuple(layer_ids)

    def _extract_link_info_key(self, module_name):
        spl_modules = module_name.split(".")

        split_id = len(spl_modules)
        for i, cur_module in enumerate(spl_modules):
            if cur_module.isdigit():
                split_id = i + 1

        return ".".join(spl_modules[:split_id])

    def _wrap_front_ds(self, module: nn.Module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        layer_id = self._extract_layer_id(module_name)
        role = self._get_module_role(module_name)

        num_experts = self.num_experts[role][layer_id]

        gate = TopKGate(
            in_features,
            num_experts,
            self.top_k,
            self.capacity_factor_train,
            self.capacity_factor_eval,
            self.min_capacity,
            self.noisy_policy,
            self.drop_tokens,
            self.use_rts,
        )

        expert_parallel_residual, expert_parallel_residual_mix = None, None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)
            expert_parallel_residual_mix = nn.Linear(in_features, 2)

        if layer_id not in self.link_info:
            self.link_info[layer_id] = dict()
        # link_info_k = self._extract_link_info_key(module_name)
        # if link_info_k not in self.link_info:
        #    self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(num_experts)
        experts = Experts(module, num_local_experts)

        _update_module_arguments(
            module=module,
            ep_context=self.ep_context,
            # link_info=self.link_info[link_info_k],
            link_info=self.link_info[layer_id],
            gate=gate,
            in_features=in_features,
            out_features=out_features,
            front_experts=experts,
            ep_group=ep_info.ep_group,
            ep_size=ep_info.ep_size,
            num_local_experts=num_local_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            expert_parallel_residual_mix=expert_parallel_residual_mix,
        )

    def _wrap_behind_ds(self, module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        layer_id = self._extract_layer_id(module_name)
        role = self._get_module_role(module_name)

        num_experts = self.num_experts[role][layer_id]

        expert_parallel_residual = None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)

        if layer_id not in self.link_info:
            self.link_info[layer_id] = dict()
        # link_info_k = self._extract_link_info_key(module_name)
        # if link_info_k not in self.link_info:
        #    self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(num_experts)
        experts = Experts(module, num_local_experts)

        _update_module_arguments(
            module,
            ep_context=self.ep_context,
            link_info=self.link_info[layer_id],
            # link_info=self.link_info[link_info_k],
            in_features=in_features,
            out_features=out_features,
            behind_experts=experts,
            ep_size=ep_info.ep_size,
            ep_group=ep_info.ep_group,
            num_local_experts=num_local_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
        )

    def _get_ep_size_param_dict(self):
        ep_size_param_dict = dict()
        for param in self.model.parameters():
            if not hasattr(param, "ep_info"):
                ep_size = 1
            else:
                ep_size = param.ep_info.ep_size

            if ep_size not in ep_size_param_dict:
                ep_size_param_dict[ep_size] = []

            ep_size_param_dict[ep_size].append(param)

        return ep_size_param_dict

    def _sync_ep_model_param(self):
        ep_info_dict = self.ep_context.parallel_info_dict
        if self.ep_context.has_setup and len(ep_info_dict) > 0:
            param_dict = self._get_ep_size_param_dict()
            if 1 in param_dict:
                _, ep_info = self.ep_context.get_info(1)
                dp_group = ep_info.get_dp_group()
                src_rank = ep_info.get_dp_group_ranks()[0]

                for param in param_dict[1]:
                    dist.broadcast(
                        param,
                        src=src_rank,
                        group=dp_group,
                    )

            for ep_size in param_dict:
                if ep_size != 1 and ep_size != self.ep_context.world_size:
                    _, ep_info = self.ep_context.get_info(ep_size)
                    src_rank = dist.get_rank(ep_info.ep_group)
                    for param in param_dict[ep_size]:
                        dist.broadcast(
                            param, src=src_rank, group=param.ep_info.dp_group
                        )
