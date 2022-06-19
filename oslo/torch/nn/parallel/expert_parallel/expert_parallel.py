import copy
import math

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
    Top1Router,
    Top2Router,
    FP32LinearGate,
    TopKGate,
)
from oslo.torch.nn.parallel.expert_parallel.layers import (
    ExpertParallelFrontBlock,
    ExpertParallelFrontBlockDS,
)
from oslo.torch.nn.parallel.expert_parallel.layers import (
    ExpertParallelBehindBlock,
    ExpertParallelBehindBlockDS,
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
        num_experts: int = 0,
        top_k: int = 2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
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

        self.num_experts = (
            num_experts if num_experts > 0 else self.ep_context.get_world_size()
        )

        self.use_residual = use_residual
        if use_residual is None:
            self.use_residual = True if top_k == 1 else False

        if noisy_policy is None:
            noisy_policy = "Jitter" if use_residual else "RSample"

        self.router_args = {
            "ep_context": self.ep_context,
            "capacity_factor_train": capacity_factor_train,
            "capacity_factor_eval": capacity_factor_eval,
            "min_capacity": min_capacity,
            "noisy_func": UniformNoiseSampler()
            if noisy_policy == "Jitter"
            else NormalNoiseSampler(num_experts),
            "drop_tks": drop_tokens,
        }
        self.router_cls = Top2Router
        self.top_k = top_k
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.noisy_policy = noisy_policy
        self.use_rts = use_rts
        self.drop_tokens = drop_tokens

        if top_k == 1:
            self.router_cls = Top1Router
            self.router_args["select_policy"] = (
                select_policy if select_policy is not None else "first"
            )

        if is_huggingface_model(model):
            mapping = _ExpertParallelMappingForHuggingFace().get_mapping(model)
        else:
            assert (
                mapping is not None
            ), "`mapping` must be input if the model is not huggingface model."
            mapping = mapping.get_mapping(model)

        self.expert_parallel_mapping = ExpertParallelMapping(mapping)

        self.link_info = dict()

        self._parallelize()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def _parallelize(self):
        self._parallelize_module()
        self.to(self.device)
        self._sync_ep_model_param()

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
                # module.__class__ = ExpertParallelFrontBlock
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
                # module.__class__ = ExpertParallelBehindBlock
                module.__class__ = ExpertParallelBehindBlockDS

        return

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

        gate = TopKGate(
            in_features,
            self.num_experts,
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

        link_info_k = self._extract_link_info_key(module_name)
        if link_info_k not in self.link_info:
            self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)
        experts = Experts(module, num_local_experts)

        _update_module_arguments(
            module=module,
            ep_context=self.ep_context,
            link_info=self.link_info[link_info_k],
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

    def _wrap_front(self, module: nn.Module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        expert_parallel_gate = FP32LinearGate(in_features, self.num_experts)
        expert_parallel_router = self.router_cls(**self.router_args)

        expert_parallel_residual, expert_parallel_residual_mix = None, None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)
            expert_parallel_residual_mix = nn.Linear(in_features, 2)

        # Add Cur Module's Link Info
        link_info_k = self._extract_link_info_key(module_name)
        if link_info_k not in self.link_info:
            self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)
        _update_module_arguments(
            module=module,
            ep_context=self.ep_context,
            in_features=in_features,
            out_features=out_features,
            num_experts=self.num_experts,
            expert_parallel_gate=expert_parallel_gate,
            expert_parallel_router=expert_parallel_router,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            expert_parallel_residual_mix=expert_parallel_residual_mix,
            link_info=self.link_info[link_info_k],
            num_local_experts=num_local_experts,
            ep_info=ep_info,
        )

        std = math.sqrt(0.1 / in_features)
        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(
                torch.empty(
                    num_local_experts, in_features, out_features, device=self.device
                ).contiguous()
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.weight = new_param

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(
                torch.empty(num_local_experts, 1, out_features, device=self.device)
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.bias = new_param

        for param in self.parameters():
            param.__setattr__("ep_info", ep_info)

        return module

    def _wrap_behind_ds(self, module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        expert_parallel_residual = None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)

        link_info_k = self._extract_link_info_key(module_name)
        if link_info_k not in self.link_info:
            self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)
        experts = Experts(module, num_local_experts)

        _update_module_arguments(
            module,
            ep_context=self.ep_context,
            link_info=self.link_info[link_info_k],
            in_features=in_features,
            out_features=out_features,
            behind_experts=experts,
            ep_size=ep_info.ep_size,
            ep_group=ep_info.ep_group,
            num_local_experts=num_local_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
        )

    def _wrap_behind(self, module, module_name: str, reversed: bool):
        out_features, in_features = module.weight.size()
        if reversed:
            out_features, in_features = in_features, out_features

        expert_parallel_residual = None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)

        # Add Cur Module's Link Info
        link_info_k = self._extract_link_info_key(module_name)
        if link_info_k not in self.link_info:
            self.link_info[link_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)

        _update_module_arguments(
            module=module,
            ep_context=self.ep_context,
            in_features=in_features,
            out_features=out_features,
            num_experts=self.num_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            link_info=self.link_info[link_info_k],
            num_local_experts=num_local_experts,
            ep_info=ep_info,
        )

        std = math.sqrt(0.1 / in_features)
        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(
                torch.empty(
                    num_local_experts, in_features, out_features, device=self.device
                ).contiguous()
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.weight = new_param

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(
                torch.empty(num_local_experts, 1, out_features, device=self.device)
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.bias = new_param

        for param in self.parameters():
            param.__setattr__("ep_info", ep_info)

        return module

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
