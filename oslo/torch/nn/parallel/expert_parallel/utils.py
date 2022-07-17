import torch
import torch.nn.functional as F
from torch import Tensor

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
        from ._ops import oslo_expert_parallel_cuda

        return oslo_expert_parallel_cuda.cumsum_sub_one(inputs)
    else:
        return torch.cumsum(inputs, dim=0) - 1


def auto_cast_softmax(logit: torch.Tensor, dim: int):
    if logit.dtype != torch.float32:
        logit = logit.float()

    return F.softmax(logit, dim=dim)


class _ForceFP32Parameter(torch.nn.Parameter):
    def half(self):
        return self.data


class NormalNoiseSampler:
    """
    A Class to sample a noisy mask for logit tensor .
    All noise is generated from a normal distribution : N(0, 1 / num_experts^2)`
    Args:
        num_experts (int): the number of experts
    """

    def __init__(self, num_experts: int):
        device = get_current_device()

        mean = torch.tensor(0.0, device=device)
        std = torch.tensor(1.0 / num_experts**2, device=device)

        self.popl = torch.distributions.normal.Normal(loc=mean, scale=std).rsample

    def __call__(self, inputs: torch.Tensor):
        noise = self.popl(inputs.shape)

        return inputs + noise


class UniformNoiseSampler:
    """
    A Class to sample a noisy mask for logit tensor
    All noise is generated from a uniform distribution : uniform(1.0 - eps, 1.0 + eps)
    Args:
        eps (float, optional): Epsilon in generator, defaults 1e-2.
    """

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
