import math

import torch.distributed as dist

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)


class _TensorParallel3DInputGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for input of 3D tensor parallelism.

    Args:
        num_group (int): The number of all tensor groups
        cubic_dim (int): cubic dim of 3D tensor parallelism
    """

    def __init__(self, num_group: int, cubic_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.cubic_dim = cubic_dim

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_3D_INPUT

        for h in range(self.num_group):
            for i in range(self.cubic_dim):
                for k in range(self.cubic_dim):
                    ranks = [
                        h * self.cubic_dim**3
                        + i
                        + self.cubic_dim * (j + self.cubic_dim * k)
                        for j in range(self.cubic_dim)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo")
                        if dist.get_backend() != "gloo"
                        else group
                    )

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


class _TensorParallel3DWeightGroupInitializer(ProcessGroupInitializer):
    def __init__(self, num_group: int, cubic_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.cubic_dim = cubic_dim

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_3D_WEIGHT

        for h in range(self.num_group):
            for k in range(self.cubic_dim):
                for j in range(self.cubic_dim):
                    ranks = [
                        h * self.cubic_dim**3
                        + i
                        + self.cubic_dim * (j + self.cubic_dim * k)
                        for i in range(self.cubic_dim)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo")
                        if dist.get_backend() != "gloo"
                        else group
                    )

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


class _TensorParallel3DOutputGroupInitializer(ProcessGroupInitializer):
    def __init__(self, num_group: int, cubic_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.cubic_dim = cubic_dim

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_3D_OUTPUT

        for h in range(self.num_group):
            for i in range(self.cubic_dim):
                for j in range(self.cubic_dim):
                    ranks = [
                        h * self.cubic_dim**3
                        + i
                        + self.cubic_dim * (j + self.cubic_dim * k)
                        for k in range(self.cubic_dim)
                    ]
                    group = dist.new_group(ranks)
                    group_cpu = (
                        dist.new_group(ranks, backend="gloo")
                        if dist.get_backend() != "gloo"
                        else group
                    )

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


class TensorParallel3DGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.cubic_dim = round(math.pow(self.tensor_parallel_size, 1 / 3))
        assert (
            self.tensor_parallel_size == self.cubic_dim**3
        ), f"3D cubic dim ({self.cubic_dim}) if not cube root of tensor parallel size ({self.tensor_parallel_size})"

        self.input_initializer = _TensorParallel3DInputGroupInitializer(
            self.num_group,
            self.cubic_dim,
            *args,
            **kwargs,
        )
        self.weight_initializer = _TensorParallel3DWeightGroupInitializer(
            self.num_group,
            self.cubic_dim,
            *args,
            **kwargs,
        )
        self.output_initializer = _TensorParallel3DOutputGroupInitializer(
            self.num_group,
            self.cubic_dim,
            *args,
            **kwargs,
        )

    def init_dist_group(self):
        return [
            self.input_initializer.init_dist_group(),
            self.weight_initializer.init_dist_group(),
            self.output_initializer.init_dist_group(),
        ]
