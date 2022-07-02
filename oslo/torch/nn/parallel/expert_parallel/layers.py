import math
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from torch.nn import Module, ModuleList
from torch.distributed import ProcessGroup

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._seed.helper import seed

from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext
from oslo.torch.nn.parallel.expert_parallel._ops import AllToAll, EPDispatch, EPCombine
from oslo.torch.nn.parallel.expert_parallel.utils import (
    get_current_device,
    cum_sum_d0_minus_one,
)
from oslo.torch.nn.parallel.expert_parallel.experts import Experts
from oslo.torch.nn.parallel.expert_parallel.utils import (
    _ForceFP32Parameter,
    auto_cast_softmax,
)
from oslo.torch.nn.parallel.expert_parallel.utils import (
    UniformNoiseSampler,
    NormalNoiseSampler,
)

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

uniform_map: Dict[torch.device, Callable] = dict()
gumbel_map: Dict[torch.device, Callable] = dict()
exp_selection_uniform_map: Dict[torch.device, Callable] = dict()

USE_EINSUM = True


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    if epsilon == 0:
        return x

    uniform = uniform_map.get(device, None)
    if uniform is None:
        low = torch.tensor(1.0 - epsilon, device=device)
        high = torch.tensor(1.0 + epsilon, device=device)
        uniform = torch.distributions.uniform.Uniform(low=low, high=high).rsample
        uniform_map[device] = uniform

    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device, None)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
        gumbel_map[device] = gumbel

    return gumbel()


# paper : https://arxiv.org/pdf/2006.16668.pdf
# Einseutine Sum Dimensions
#     - g = group
#     - s = sequence
#     - e = expert
#     - m = model
#     - c = capacity
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == "s,se->se":
        return a.reshape(a.shape[0], -1) * b
    elif rule == "se,sc->sec":
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == "se,se->s":
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == "sec,sm->ecm":
        s, e, c = a.shape[:3]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == "sec,ecm->sm":
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == "ks,ksm->sm":
        k, s, m = b.shape[:3]

        a = a.t().unsqueeze(1)
        # |a| = (k, s) -> |a.t()| = (s, k) -> |a.t().unsqueeze(1)| = (s, 1, k)
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # |b| = (k, s, m) -> |b.reshape(k, -1)| = (k, s*m) -> |b.reshape(k, -1).t()| = (s*m, k)
        # -> |b.reshape(k, -1).t().reshape(s, m, k)| = (s, m, k)

        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.
@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # |gates| = (S, E)
    num_tokens, num_experts = gates.shape[:2]

    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int 64 when converting torch.topk but it doesn't
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)

    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


# TODO : Change the code based on tutel to code based on cuda-kernel
def top1_gating(
    logits: Tensor,
    capacity_factor: float,
    min_capacity: int,
    noisy_gate_policy: Optional[str] = None,
    drop_tokens: bool = True,
    use_rts: bool = True,
    use_kernel: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # |logits| = (num_tokens, num_experts)

    # For exploration, add noise to logits.
    # In this case, since noisy_gate_policy is different,
    # do not multiply noise with inputs in the gate.
    if noisy_gate_policy == "RSample":
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
        # |logits_w_noise| = (num_tokens, num_experts)

    # Calculate conditional probability for each token to be dispatched to experts
    gates = F.softmax(logits, dim=1)
    # |gates| = (num_tokens, num_experts)

    # Calculate the number of tokens that each expert can contain
    capacity = _capacity(
        gates, torch.tensor(capacity_factor), torch.tensor(min_capacity)
    )

    # Get the index of expert for each token to be dispatched
    indices = torch.argmax(
        logits_w_noise if noisy_gate_policy == "RSample" else gates, dim=1
    )
    # |indices| = (num_tokens, )

    # Binarize the index of expert for each token to be dispatched
    num_experts = int(gates.shape[1])
    mask = F.one_hot(indices, num_classes=num_experts)
    # |mask| = (num_tokens, num_experts)

    # Calculate the number of tokens dispatched to each expert
    exp_counts = torch.sum(mask, dim=0).detach().to("cpu")
    # |exp_counts| = (num_experts, )

    # Share the maximum capacity across all devices in the world
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
        capacity = new_capacity

    # Calculate loss for training
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # For exploration of token selection, multiply noise with mask
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device, None)
        if uniform is None:
            low = torch.tensor(0.0, device=logits.device)
            high = torch.tensor(1.0, device=logits.device)
            uniform = torch.distributions.uniform.Uniform(low=low, high=high).rsample
            exp_selection_uniform_map[logits.device] = uniform
        mask_rand = mask * uniform(mask.shape)
    else:
        mask_rand = mask
    # |mask_rand| = (num_tokens, num_experts)

    assert (
        logits.shape[0] >= min_capacity
    ), "No, of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size"

    top_idx = _top_idx(mask_rand, capacity)
    # |top_idx| = (num_tokens, 1)

    mask = mask * torch.zeros_like(mask).scatter_(0, top_idx, 1)
    # |mask| = (num_tokens, num_experts)

    locations = torch.cumsum(mask, dim=0) - 1
    # |locations| = (num_tokens, )

    # Get the location of token in the buffer allocated to each expert
    # If some tokens drop out, then clear all values of that token
    locations_s = torch.sum(locations * mask, dim=1)
    # |locations_s| = (num_tokens, )

    # Only leave the conditional probability for the token to be dispatched
    # and clear other conditional probabilities for that token
    mask_float = mask.float()
    gates = gates * mask_float
    # |mask_float| = (num_tokens, num_experts)
    # |gates| = (num_tokens, num_experts)

    # Binarize the location of token in the buffer allocated to each expert
    locations_sc = _one_hot_to_float(locations_s, capacity)
    # |location_sc| = (num_tokens, num_experts)

    combine_weights = einsum("se,sc->sec", gates, locations_sc)
    dispatch_mask = combine_weights.bool()
    # |combine_weights| = (num_tokens, num_experts, capacity)
    # |dispatch_mask| = (num_tokens, num_experts, capacity)

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2_gating(
    logits: Tensor, capacity_factor: float, min_capacity: int, use_kernel: bool
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # |logits| = (num_tokens, num_experts)

    # Calculate conditional probability for each token to be dispatched to experts
    gates = F.softmax(logits, dim=1)
    # |gates| = (num_tokens, num_experts)

    # Calculate the number of tokens that each expert can contain
    capacity = _capacity(
        gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity)
    )

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # |indices1_s| = (token_num, )
    # |mask1| = (token_num, num_experts)

    # TODO : Analyze the following code
    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.shape)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    # |logits_w_noise| = (num_tokens, num_experts)
    # |logits_except1| = (num_tokens, num_experts)

    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    # |indices2_s| = (num_tokens, 1)
    # |mask2| = (num_tokens, num_experts)

    # Get the location of token in the buffer allocated to each expert
    # If some tokens drop out, then clear all values of that token
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    # |locations1| = (num_tokens, num_experts)
    # |locations2| = (num_tokens, num_experts)

    # Calculate the number of tokens dispatched to each expert
    exp_counts = torch.sum(mask1, dim=0)
    # |exp_counts| = (num_experts, )

    # Calculate loss for training
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # Drop out tokens spilling out from the buffer
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)
    # |mask1| = (num_tokens, num_experts)
    # |mask2| = (num_tokens, num_experts)

    # Get the location of token in the buffer allocated to each expert
    # If some tokens drop out, then clear all values of that token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    # |locations1_s| = (num_tokens, num_experts)
    # |locations2_s| = (num_tokens, num_experts)

    mask1_float = mask1.float()
    mask2_float = mask2.float()

    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    # |gates1_s| = (num_tokens, )
    # |gates2_s| = (num_tokens, )

    # Normalize conditional probabilities for tokens to be dispatched
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    # |denom_s| = (num_tokens, )

    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    # |gates1| = (num_tokens, num_experts)
    # |gates2| = (num_tokens, num_experts)

    # Binarize the location of token in the buffer allocated to each expert
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    # |locations1_sc| = (num_tokens, capacity)
    # |locations2_sc| = (num_tokens, capacity)

    # Calculate combine weights and dispatch mask for output of different experts but of same tokens
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    # |combine1_sec| = (num_tokens, num_experts, capacity)
    # |combine2_sec| = (num_tokens, num_experts, capacity)

    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    # |combine_weights| = (num_tokens, num_experts, capacity)
    # |dispatch_mask| = (num_tokens, num_experts, capacity)

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(Module):

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 1,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
    ) -> None:
        super().__init__()

        if k != 1 and k != 2:
            raise ValueError("Only top-1 and top-2 gatings are supported")
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = k

        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity

        self.noisy_gate_policy = noisy_gate_policy

        self.drop_tokens = drop_tokens
        self.use_rts = use_rts

    def forward(
        self, inputs: torch.Tensor, use_kernel: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # |inputs| = (token_num = batch_size * sent_len, d_model)

        # For stability of training process, make tensor 32 bit float type
        # of gate's weights and input
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        input_fp32 = inputs.float()

        # For exploration, multiply noise with input
        if self.noisy_gate_policy == "Jitter" and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=inputs.device)

        # Calculate gating score for each token
        logits = self.wg(input_fp32)
        # |logits| = (tokens_num, experts_num)

        if self.k == 1:
            gate_output = top1_gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens,
                self.use_rts,
                use_kernel,
            )
        else:
            gate_output = top2_gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
            )
        # gate_output
        #   gate_output[0] = auxiliary loss
        #   gate_output[1] = weights to combine the outputs of different experts but based on same token
        #   gate_output[2] = mask to dispatch tokens for each expert
        #   gate_output[3] = number of tokens to be dispatched for each expert

        return gate_output


# TODO : Implement kernel optimized code
class ExpertParallelFrontBlockDS(Module):
    def __init__(
        self,
        ep_context: ExpertParallelContext,
        link_info: dict,
        in_features: int,
        out_features: int,
        gate: Module,
        experts: Experts,
        ep_size: int,
        num_local_experts: int,
        use_residual: bool = False,
        expert_parallel_residual: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.ep_context = ep_context
        self.link_info = link_info

        self.gate = gate

        self.in_features = in_features
        self.out_features = out_features

        self.front_experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts

        # Create Residual
        self.use_residual = use_residual
        self.expert_parallel_residual, self.expert_parallel_residual_mix = None, None
        if use_residual:
            assert (
                expert_parallel_residual is not None
            ), "If you want to use residual moe, then you must give residual instance"
            self.expert_parallel_residual = expert_parallel_residual

            self.expert_parallel_residual_mix = nn.Linear(in_features, 2)

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def residual_forward(self, inputs: Tensor):
        residual_inter = self.expert_parallel_residual(inputs)
        # |residual_inter| = (sent_len, batch_size, out_features)

        residual_weight = self.expert_parallel_residual_mix(inputs)
        # residual_weight = F.softmax(residual_weight, dim=-1)
        residual_weight = F.softmax(residual_weight, dim=1)
        # |residual_weight| = (sent_len, batch_size, 2)

        return residual_inter, residual_weight

    def forward(self, inputs: Tensor):
        # |inputs| = (batch_size, sent_len, d_model)
        #         or (sent_len, batch_size, d_model)

        # Implement Algorithm 2 from GShard Paper
        d_model = inputs.shape[-1]

        # Initial Implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwarg['group_size'] if 'group_size' in kwargs.size() else 1
        # -> this is not valid for oslo's context because we cannot manipulate inputs of each intra-modules
        self.link_info["inputs_shape"] = inputs.shape
        reshaped_input = inputs.reshape(-1, d_model)
        # |reshaped_input| = (num_tokens = batch_size * sent_len, d_model)

        # TODO : Implement the code based on kernel optimization
        # Calculate information to dispatch tokens
        l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(
            reshaped_input
        )
        self.link_info["combine_weights"] = combine_weights
        # |combine_weights| = (num_tokens, num_experts, capacity)
        # |dispatch_mask| = (num_tokens, num_experts, capacity)

        dispatched_input = einsum(
            "sec,sm->ecm", dispatch_mask.type_as(inputs), reshaped_input
        )
        # |dispatched_input| = (num_experts, capacity, d_model)

        # Dispatch tokens to each expert
        dispatched_input = AllToAll.apply(self.ep_group, dispatched_input)
        self.link_info["a2a_shape"] = dispatched_input.shape
        # |dispatched_input| = (num_experts, capacity, in_features)

        # Reshape tokens for convenience of experts' forward operation
        dispatched_input = dispatched_input.reshape(
            self.ep_size, self.num_local_experts, -1, self.in_features
        )
        # |dispatched_input| = (ep_size, num_local_experts, capacity, in_features)

        # TODO : Think about the needs to recover the shape of experts' input (a.k.a expert_shape)
        expert_output = self.front_experts(dispatched_input)
        # |expert_output| = (ep_size, num_local_experts, capacity, out_features)

        self.link_info["front_output_shape"] = expert_output.shape
        output = expert_output.reshape(-1, self.link_info["front_output_shape"][-1])
        # |output| = (ep_size * num_local_experts * capacity, out_features)

        if self.use_residual:
            residual_inter, self.link_info["residual_weight"] = self.residual_forward(
                inputs
            )

            self.link_info["residual_inter_shape"] = residual_inter.shape
            residual_inter = residual_inter.reshape(
                -1, self.link_info["residual_inter_shape"][-1]
            )
            # |residual_inter| = (sent_len * batch_size, out_features)

            output = torch.cat([output, residual_inter], dim=0)
            # |output| = (ep_size * num_local_experts * capacity + sent_len * batch_size, out_features)

        return output


# TODO : Implement Kernel Optimized Code
class ExpertParallelBehindBlockDS(Module):
    def __init__(
        self,
        ep_context: ExpertParallelContext,
        link_info: dict,
        in_features: int,
        out_features: int,
        experts: Experts,
        ep_size: int,
        num_local_experts: int,
        use_residual: bool = False,
        expert_parallel_residual: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.ep_context = ep_context
        self.link_info = link_info

        self.in_features = in_features
        self.out_features = out_features

        self.behind_experts = experts
        # TODO : Need to implement initialize ep_group
        self.ep_group = None
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts

        self.use_residual = use_residual
        self.expert_parallel_residual = None
        if use_residual:
            self.expert_parallel_residual = expert_parallel_residual

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, inputs):
        # |inputs| =  (ep_size * num_local_experts * capacity, in_features) if not use_residual
        #         or (ep_size * num_local_experts * capacity + sent_len * batch_size, out_features) if use_residual

        dim0, dim1, dim2, _ = self.link_info["front_output_shape"]
        front_output = inputs[: dim0 * dim1 * dim2].reshape(
            self.link_info["front_output_shape"]
        )
        # |front_output| = (ep_size, num_local_experts, capacity, out_features)

        expert_output = self.behind_experts(front_output)
        expert_output = AllToAll.apply(self.ep_group, expert_output)
        # |expert_output| = (ep_size, num_local_experts, capacity, out_features)

        expert_output = expert_output.reshape(
            self.ep_size * self.num_local_experts, -1, self.out_features
        )
        # |expert_output| = (ep_size * num_local_experts, capacity, out_features)

        # Combine
        combined_output = einsum(
            "sec,ecm->sm",
            self.link_info["combine_weights"].type_as(front_output),
            expert_output,
        )
        # |combined_output| = (token_num = batch_size * sent_len, out_features)

        output = combined_output.reshape(self.link_info["inputs_shape"])
        # |output| = (sent_len, batch_size, out_features)
        #         or (batch_size, sent_len, out_features)

        # Resiudal
        if self.use_residual:
            residual_inter = inputs[dim0 * dim1 * dim2 :].reshape(
                self.link_info["residual_inter_shape"]
            )
            # |residual_inter| = (sent_len, batch_size, in_features)
            #                 or (batch_size, sent_len, in_features)
            residual_output = self.expert_parallel_residual(residual_inter)
            # |residual_output| = (sent_len, batch_size, out_features)

            output = (
                output * self.link_info["residual_weight"][..., 0:1]
                + residual_output * self.link_info["residual_weight"][..., 1:]
            )
            # |output| = (sent_len, batch_size, out_features)

        return output
