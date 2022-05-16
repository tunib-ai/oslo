import copy
import math

import torch
import torch.nn as nn

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._seed.helper import seed

from oslo.torch.distributed.parallel_context import ParallelContext
from oslo.torch.nn.parallel.utils import ParallelWrapper
from oslo.torch.nn.parallel.utils import is_huggingface_model, _update_module_arguments

from oslo.transformers.mapping_utils import _ExpertParallelMappingForHuggingFace

from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext
from oslo.torch.nn.parallel.expert_parallel.mapping import ExpertParallelMapping

from oslo.torch.nn.parallel.expert_parallel.layers import (
    Top1Router,
    Top2Router,
    FP32LinearGate,
)
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelFrontBlock
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelBehindBlock

from oslo.torch.nn.parallel.expert_parallel._ops import OSLO_EP_KERNEL_FLAG

from oslo.torch.nn.parallel.expert_parallel.utils import (
    UniformNoiseSampler,
    NormalNoiseSampler,
)


class ExpertParallel(ParallelWrapper):
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
        drop_tks: bool = True,
        use_residual: bool = None,
    ):
        super().__init__()

        self.model = model

        use_kernel_optim = OSLO_EP_KERNEL_FLAG and use_kernel_optim
        self.ep_context = ExpertParallelContext(parallel_context, use_kernel_optim)
        self.ep_context.setup(parallel_context.seed)

        self.device = torch.cuda.current_device()

        self.num_experts = (
            num_experts if num_experts > 0 else self.ep_context.get_world_size()
        )

        if use_residual is None:
            self.use_residual = True if top_k == 1 else False

        if noisy_policy is None:
            noisy_policy = "Jitter" if use_residual else "Gaussian"
        self.router_args = {
            "ep_context": self.ep_context,
            "capacity_factor_train": capacity_factor_train,
            "capacity_factor_eval": capacity_factor_eval,
            "min_capacity": min_capacity,
            "noisy_func": UniformNoiseSampler()
            if noisy_policy == "Jitter"
            else NormalNoiseSampler(num_experts),
            "drop_tks": drop_tks,
        }
        self.router_cls = Top2Router
        if top_k == 1:
            self.router_cls = Top1Router
            self.router_args["select_policy"] = (
                select_policy if select_policy is not None else "first"
            )

        if is_huggingface_model(model):
            mapping = _ExpertParallelMappingForHuggingFace().get_mapping(model)
        else:
            raise ValueError(
                "`mapping` must be input if the model is not huggingface model."
            )
        self.expert_parallel_mapping = ExpertParallelMapping(mapping)

        self.combine_info = dict()

        self._parallelize()

    @torch.no_grad()
    def _parallelize(self):
        self._parallelize_module()

    def _parallelize_module(self):
        for module_name, module in self.model.named_modules():
            if self.expert_parallel_mapping.is_front_parallel(self.model, module_name):
                self._wrap_front(module, module_name)
                module.__class__ = ExpertParallelFrontBlock
            elif self.expert_parallel_mapping.is_behind_parallel(
                self.model, module_name
            ):
                self._wrap_behind(module, module_name)
                module.__class__ = ExpertParallelBehindBlock

        return

    def _extract_combine_info_key(self, module_name):
        spl_modules = module_name.split(".")

        split_id = len(spl_modules)
        for i, cur_module in enumerate(spl_modules):
            if cur_module.isdigit():
                split_id = i + 1

        return ".".join(spl_modules[:split_id])

    def _wrap_front(self, module: nn.Module, module_name: str):
        out_features, in_features = module.weight.size()

        expert_parallel_gate = FP32LinearGate(in_features, self.num_experts)
        expert_parallel_router = self.router_cls(**self.router_args)

        expert_parallel_residual, expert_parallel_residual_mix = None, None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)
            expert_parallel_residual_mix = nn.Linear(in_features, 2)

        # Add Cur Module's Combine Info
        combine_info_k = self._extract_combine_info_key(module_name)
        if combine_info_k not in self.combine_info:
            self.combine_info[combine_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)
        _update_module_arguments(
            module=module,
            parallel_context=self.ep_context,
            in_features=in_features,
            out_features=out_features,
            num_experts=self.num_experts,
            expert_parallel_gate=expert_parallel_gate,
            expert_parallel_router=expert_parallel_router,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            expert_parallel_residual_mix=expert_parallel_residual_mix,
            combine_info=self.combine_info[combine_info_k],
            num_local_experts=num_local_experts,
            ep_info=ep_info,
        )

        std = math.sqrt(0.1 / in_features)
        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(
                torch.empty(num_local_experts, in_features, out_features).contiguous()
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            self.weight = new_param

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(torch.empty(num_local_experts, 1, out_features))
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.bias = new_param

        for param in self.parameters():
            param.__setattr__("ep_info", ep_info)

        return module

    def _wrap_behind(self, module, module_name):
        out_features, in_features = module.weight.size()

        expert_parallel_residual = None
        if self.use_residual:
            expert_parallel_residual = copy.deepcopy(module)

        # Add Cur Module's Combine Info
        combine_info_k = self._extract_combine_info_key(module_name)
        if combine_info_k not in self.combine_info:
            self.combine_info[combine_info_k] = dict()

        num_local_experts, ep_info = self.ep_context.get_info(self.num_experts)

        _update_module_arguments(
            module=module,
            ep_context=self.ep_context,
            in_features=in_features,
            out_features=out_features,
            num_experts=self.num_experts,
            use_residual=self.use_residual,
            expert_parallel_residual=expert_parallel_residual,
            combine_info=self.combine_info[combine_info_k],
            num_local_experts=num_local_experts,
            ep_info=ep_info,
        )

        std = math.sqrt(0.1 / in_features)
        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(
                torch.empty(num_local_experts, in_features, out_features).contiguous()
            )
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.weight = new_param

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(torch.empty(num_local_experts, 1, out_features))
            with seed(ParallelMode.TENSOR):
                nn.init.trunc_normal_(new_param, std=std)
            module.bias = new_param

        for param in self.parameters():
            param.__setattr__("ep_info", ep_info)

        return module
