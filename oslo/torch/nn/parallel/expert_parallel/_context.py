import torch
import torch.distributed as dist

from oslo.torch.distributed._seed.helper import moe_set_seed


def _check_sanity(parallel_context):
    if (
        parallel_context.tensor_parallel_size > 1
        or parallel_context.pipeline_parallel_size > 1
        or parallel_context.sequence_parallel_size > 1
    ):
        raise NotImplementedError(
            "Expert parallelism is not compatible with "
            "tensor or pipeline or sequence parallel at present."
        )


class ExpertParallelInfo(object):
    """
    A class to describe information expert parallelization and expert data parallelization

    Args:
            ep_size: the number of nodes in expert parallel group
            dp_size: the number of nodes in expert data parallel group
            parallel_context: global parallel context
    """

    def __init__(self, ep_size, dp_size, parallel_context):

        self.ep_size = ep_size
        self.dp_size = dp_size
        self.ep_group = None
        self.dp_group = None

        self.ep_group_ranks = None
        self.dp_group_ranks = None

        # Create expert parallel group
        rank = parallel_context.get_global_rank()
        for i in range(dp_size):
            ranks = [i * ep_size + j for j in range(ep_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                self.ep_group = group
                self.ep_group_ranks = ranks
                self.ep_local_rank = ranks.index(rank)

        # Create expert data parallel group
        for j in range(ep_size):
            ranks = [i * ep_size + j for i in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                self.dp_group = group
                self.dp_group_ranks = ranks
                self.dp_local_rank = ranks.index(rank)

    def get_dp_group(self):
        return self.dp_group

    def get_ep_group(self):
        return self.ep_group

    def get_dp_local_rank(self):
        return self.dp_local_rank

    def get_ep_local_rank(self):
        return self.ep_local_rank

    def get_ep_group_ranks(self):
        return self.ep_group_ranks

    def get_dp_group_ranks(self):
        return self.dp_group_ranks


class ExpertParallelContext(object):
    """
    A class to describe the Context about Expert Parallel

    Args:
            parallel_context: global parallel context
            use_kernel_optim: flag to use kernel optimization
    """

    def __init__(self, parallel_context, use_kernel_optim):
        self.world_size = parallel_context.expert_parallel_size
        self.parallel_context = parallel_context
        self.use_kernel_optim = use_kernel_optim
        self.min_dp_size = 1
        self.aux_loss = None

        self.has_setup = False
        self._parallel_info_dict = dict()

    @property
    def parallel_info_dict(self):
        return self._parallel_info_dict

    @property
    def is_initialized(self):
        return self.has_setup

    def setup(self, seed: int):
        """
        Set base information about expert parallel context

        Args:
            seed: random seed value for expert parallel
        """

        assert (
            not self.is_initialized
        ), "MoE distributed context shouldn't be set up again"
        _check_sanity(self.parallel_context)

        assert (
            self.world_size % self.parallel_context.expert_parallel_size == 0
        ), "Maximum expert parallel size must be a factor of the number of GPUs"

        self.min_dp_size = self.world_size // self.parallel_context.expert_parallel_size
        moe_set_seed(self.parallel_context, seed)
        self.has_setup = True

    def get_info(self, num_experts: int):
        """
        If there is no information about given num_experts, create, save and return the information about expert parallel.
        Otherwise, just return the information about expert parallel

        Args:
            num_experts: the number of experts

        Returns:
            num_local_experts: the number of local experts
        """

        gt_flag = (
            num_experts % self.parallel_context.expert_parallel_size == 0
        )  # check whether num_experts is greater
        lt_flag = (
            self.parallel_context.expert_parallel_size % num_experts == 0
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
        dp_size = (
            1 if gt_flag else self.parallel_context.expert_parallel_size // num_experts
        )
        ep_size = self.parallel_context.expert_parallel_size // dp_size

        # Calculate the number of experts for each GPU
        num_local_experts = (
            1 if lt_flag else num_experts // self.parallel_context.expert_parallel_size
        )

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

    def get_world_size(self):
        return self.world_size
