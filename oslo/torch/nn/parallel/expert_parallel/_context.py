import torch
import torch.distributed as dist

from oslo.torch.distributed._seed.helper import moe_set_seed
from oslo.torch.distributed.parallel_mode import ParallelMode


def _check_sanity(parallel_context):
    if (
        parallel_context.tensor_parallel_size > 1
        or parallel_context.pipeline_parallel_size > 1
    ):
        raise NotImplementedError(
            "Moe is not compatible with tensor or pipeline parallel at present."
        )


class ExpertParallelInfo(object):
    def __init__(self, ep_size, dp_size, parallel_context):
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.ep_group = None
        self.dp_group = None

        if ep_size == 1:
            self.ep_group = parallel_context.get_group(ParallelMode.TENSOR)
            self.dp_group = parallel_context.get_group(ParallelMode.DATA)
            return

        if dp_size == 1:
            self.ep_group = parallel_context.get_group(ParallelMode.DATA)
            self.dp_group = parallel_context.get_group(ParallelMode.TENSOR)
            return

        rank = dist.get_rank()
        # Create expert parallel group
        for i in range(dp_size):
            ranks = [i * ep_size + j for j in range(ep_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                self.ep_group = group

        # Create data parallel group
        for j in range(ep_size):
            ranks = [i * ep_size + j for i in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                self.dp_group = group


class ExpertParallelContext(object):
    def __init__(self, parallel_context, max_ep_size):
        self.max_ep_size = max_ep_size
        self.parallel_context = parallel_context
        self.world_size = 1
        self.min_dp_size = 1
        self.aux_loss = None
        self.use_kernel_optim = True

        self.has_setup = False
        self._parallel_info_dict = dict()

    @property
    def parallel_info_dict(self):
        return self._parallel_info_dict

    @property
    def is_initialized(self):
        return self.has_setup

    def setup(self, seed: int, use_kernel_optim: bool = True):

        assert (
            not self.is_initialized
        ), "MoE distributed context shouldn't be set up again"
        _check_sanity(self.parallel_context)
        assert torch.cuda.is_available(), "MoE requires to enable CUDA first"

        self.world_size = dist.get_world_size()

        assert (
            self.world_size % self.max_ep_size == 0
        ), "Maximum expert parallel size must be a factor of the number of GPUs"
        self.min_dp_size = self.world_size // self.max_ep_size

        # Enabling kernel optimization may raise error in some cases
        # Users can close kernel optimization manually
        self.use_kernel_optim = use_kernel_optim
        moe_set_seed(seed)
        self.has_setup = True

    def get_info(self, num_experts: int):
        gt_flag = (
            num_experts % self.max_ep_size == 0
        )  # check whether num_experts is greater
        lt_flag = (
            self.max_ep_size % num_experts == 0
        )  # check whether num_experts is less

        assert gt_flag or lt_flag, (
            "Automatic experts placement dose not not support expert number"
            " is not a multiple of ep size or vice versa."
        )

        # If the number of experts is greater than maximum expert parallel size. a.k.a ep_size,
        # there are multiple experts in each GPU and each GPU has different experts
        # So it's data parallel size is 1
        # Otherwise, there is only one expert in each GPU
        # The data parallel size should be calculated
        dp_size = 1 if gt_flag else self.max_ep_size // num_experts
        ep_size = self.max_ep_size // dp_size

        # Calculate the number of experts for each GPU
        num_local_experts = 1 if lt_flag else num_experts // self.max_ep_size

        # Don't forget to multiply minimum data parallel size
        dp_size *= self.min_dp_size
        if not (ep_size in self.parallel_info_dict):
            self.parallel_info_dict[ep_size] = ExpertParallelInfo(
                ep_size, dp_size, self.parallel_context
            )

        return num_local_experts, self.parallel_info_dict[ep_size]

    def set_kernel_not_use(self):
        self.use_kernel_optim = False

    def reset_loss(self):
        self.aux_loss = 0

    def add_loss(self, loss):
        self.aux_loss += loss

    def get_loss(self):
        return self.aux_loss
