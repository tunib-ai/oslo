import math

import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed._parallel_mode import ParallelMode


class _TensorParallel2DRowGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for row dimension of 2D tensor parallelism.

    Args:
        num_group (int): The number of all tensor parallel groups
        summa_dim (int): The dimension of SUMMA
    """

    def __init__(self, num_group: int, summa_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.summa_dim = summa_dim

    def init_dist_group(self):
        """
        Initialize 2D tensor row parallel groups and assign local_ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2D_ROW

        for i in range(self.num_group):
            for j in range(self.summa_dim):
                ranks = [
                    i * self.tensor_parallel_size + j * self.summa_dim + k
                    for k in range(self.summa_dim)
                ]
                group = dist.new_group(ranks)
                group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }


class _TensorParallel2DColumnGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for column dimension of 2D tensor parallelism.

    Args:
        num_group (int): The number of all tensor parallel groups
        summa_dim (int): The dimension of SUMMA
    """

    def __init__(self, num_group, summa_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.summa_dim = summa_dim

    def init_dist_group(self):
        """
        Initialize 2D tensor column parallel groups and assign local_ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2D_COL

        for i in range(self.num_group):
            for j in range(self.summa_dim):
                ranks = [
                    i * self.tensor_parallel_size + j + k * self.summa_dim
                    for k in range(self.summa_dim)
                ]
                group = dist.new_group(ranks)
                group_cpu = dist.new_group(ranks, backend='gloo') if dist.get_backend() != 'gloo' else group

                if self.rank in ranks:
                    local_rank = ranks.index(self.rank)
                    group_world_size = len(ranks)
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "cpu_group": cpu_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }


class TensorParallel2DGroupInitializer(ProcessGroupInitializer):
    """Process group initializer for 2D tensor parallelism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.summa_dim = int(math.sqrt(self.tensor_parallel_size))

        assert (
            self.tensor_parallel_size == self.summa_dim**2
        ), "2D summa dim should equal to tensor parallel size ^ 0.5"

        self.col_initializer = _TensorParallel2DColumnGroupInitializer(
            self.num_group, self.summa_dim
        )
        self.row_initializer = _TensorParallel2DRowGroupInitializer(
            self.num_group, self.summa_dim
        )

    def init_dist_group(self):
        """
        Initialize 2D row and column parallel groups and assign local ranks and groups to each GPU.

        Returns:
            List[Dict]: list of 2d row and column parallel parallel group information
        """

        return [
            self.row_initializer.init_dist_group(),
            self.col_initializer.init_dist_group(),
        ]
