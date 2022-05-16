import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed import ParallelContext

from ._ops import OSLO_EP_KERNEL_FLAG


def get_current_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


def cum_sum_d0_minus_one(inputs: Tensor):
    dim0 = inputs.size(0)
    flag = (dim0 <= 1024) or (dim0 <= 2048 and dim0 % 2 == 0) or (dim0 % 4 == 0)
    if flag and OSLO_EP_KERNEL_FLAG:
        import oslo_expert_parallel_cuda

        return oslo_expert_parallel_cuda.cumsum_sub_one(inputs)
    else:
        return torch.cumsum(inputs, dim=0) - 1


def auto_cast_softmax(logit: torch.Tensor, dim: int):
    if logit.dtype != torch.float32:
        logit = logit.float()

    return F.softmax(logit, dim=dim)


def _get_ep_size_param_dict(model: nn.Module):
    ep_size_param_dict = dict()
    for param in model.parameters():
        if not hasattr(param, "ep_info"):
            ep_size = 1
        else:
            ep_size = param.ep_info.ep_size

        if ep_size not in ep_size_param_dict:
            ep_size_param_dict[ep_size] = []

        ep_size_param_dict[ep_size].append(param)

    return ep_size_param_dict


def _sync_ep_model_param(model: nn.Module):
    parallel_context = ParallelContext.from_torch()
    if (
        parallel_context.is_initialized(ParallelMode.DATA)
        and parallel_context.get_world_size(ParallelMode.DATA) > 1
    ):

        param_dict = _get_ep_size_param_dict(model)
        if 1 in param_dict:
            src_rank = parallel_context.get_ranks_in_group(ParallelMode.DATA)[0]

            for param in param_dict[1]:
                dist.broadcast(
                    param,
                    src=src_rank,
                    group=parallel_context.get_group(ParallelMode.DATA),
                )

        # TOD: Think About the Way to get the Expert Parallel Context
        # ep_context = ExpertParallelContext.from_torch()
        # for ep_size in param_dict:
        #    if ep_size != 1 and ep_size != ep_context.world_size:
        #        src_rank = dist.get_rank(ep_context.parallel_info_dict[ep_size].ep_group)
        #
        #         for param in param_dict[ep_size]:
        #            dist.broadcast(param, src=src_rank, group=param.ep_info.dp_group)

    return


class _ForceFP32Parameter(torch.nn.Parameter):
    def half(self):
        return self.data


class NormalNoiseSampler:
    def __init__(self, num_experts: int):
        device = get_current_device()

        mean = torch.tensor(0.0, device=device)
        std = torch.tensor(1.0 / num_experts**2, device=device)

        self.popl = torch.distributions.normal.Normal(loc=mean, scale=std).rsample

    def __call__(self, inputs: torch.Tensor):
        noise = self.popl(inputs.shape)

        return inputs + noise


class UniformNoiseSampler:
    def __init__(self, eps: float = 1e-2):
        device = get_current_device()

        lower_bound = torch.tensor(1.0 - eps, device=device)
        upper_bound = torch.tensor(1.0 + eps, device=device)

        self.popl = torch.distributions.uniform.Uniform(
            low=lower_bound, high=upper_bound
        ).rsample

    def __call__(self, inputs: torch.Tensor):
        noise = self.popl(inputs.shape)

        return inputs * noise
