import math

import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed._parallel_mode import ParallelMode


# i row, j col, k dep
class _TensorParallel2p5DRowGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for row dimension of 2.5D tensor parallelism.

    Args:
        tesseract_dim (int): The dimension of tesseract
        tesseract_dep (int): The dimension of depth
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dim = tesseract_dim
        self.tesseract_dep = tesseract_dep
        assert (
            self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep
        ), "Tensor parallel size should be depth * dim ** 2 in 2.5D parallel"

    def init_dist_group(self):
        """
        Initialize 2.5D tensor row parallel groups and assign local ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2P5D_ROW

        for h in range(self.num_group):
            for j in range(self.tesseract_dim):
                for k in range(self.tesseract_dep):
                    ranks = [
                        h * self.tensor_parallel_size
                        + i
                        + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for i in range(self.tesseract_dim)
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


class _TensorParallel2p5DColumnGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for column dimension of 2.5D tensor parallelism.

    Args:
        tesseract_dim (int): The dimension of tesseract
        tesseract_dep (int): The dimension of depth
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dim = tesseract_dim
        self.tesseract_dep = tesseract_dep
        assert (
            self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep
        ), "Tensor parallel size should be depth * dim ** 2 in 2.5D parallel"

    def init_dist_group(self):
        """
        Initialize 2.5D tensor column parallel groups and assign local ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """

        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2P5D_COL

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                for k in range(self.tesseract_dep):
                    ranks = [
                        h * self.tensor_parallel_size
                        + i
                        + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for j in range(self.tesseract_dim)
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


class _TensorParallel2p5DDepthGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for depth dimension of 2.5D tensor parallelism.

    Args:
        tesseract_dim (int): The dimension of tesseract
        tesseract_dep (int): The dimension of depth
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dim = tesseract_dim
        self.tesseract_dep = tesseract_dep
        assert (
            self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep
        ), "Tensor parallel size should be depth * dim ** 2 in 2.5D parallel"

    def init_dist_group(self):
        """
        Initialize 2.5D tensor depth parallel groups and assign local ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2P5D_DEP

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                for j in range(self.tesseract_dim):
                    ranks = [
                        h * self.tensor_parallel_size
                        + i
                        + self.tesseract_dim * (j + self.tesseract_dim * k)
                        for k in range(self.tesseract_dep)
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


# i row, j col, k dep
class _TensorParallel2p5DXZGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for time depth dimension of 2.5D tensor parallelism.

    Args:
        tesseract_dim (int): The dimension of tesseract
        tesseract_dep (int): The dimension of depth
    """

    def __init__(self, tesseract_dim: int, tesseract_dep: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dep = tesseract_dep
        self.tesseract_dim = tesseract_dim
        assert (
            self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep
        ), "Tensor parallel size should be depth * dim ** 2 in 2.5D parallel"

    def init_dist_group(self):
        """
        Initialize 2.5D tensor column X depth parallel groups and assign local ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR_2P5D_XZ

        for h in range(self.num_group):
            for i in range(self.tesseract_dim):
                ranks = [
                    h * self.tensor_parallel_size
                    + i
                    + self.tesseract_dim * (j + self.tesseract_dim * k)
                    for k in range(self.tesseract_dep)
                    for j in range(self.tesseract_dim)
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


class TensorParallel2p5DGroupInitializer(ProcessGroupInitializer):
    """Process group initializer for 2.5D tensor parallelism."""

    def __init__(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = self.world_size // self.tensor_parallel_size
        self.tesseract_dim = int(math.sqrt(self.tensor_parallel_size / depth))
        self.tesseract_dep = depth

        assert (
            self.tensor_parallel_size == self.tesseract_dim ** 2 * self.tesseract_dep
        ), "2.5D tesseract dim should equal to (tensor parallel size / tesseract dep) ^ 0.5"

        self.col_initializer = _TensorParallel2p5DColumnGroupInitializer(
            self.tesseract_dim, self.tesseract_dep, *args, **kwargs
        )
        self.row_initializer = _TensorParallel2p5DRowGroupInitializer(
            self.tesseract_dim, self.tesseract_dep, *args, **kwargs
        )
        self.dep_initializer = _TensorParallel2p5DDepthGroupInitializer(
            self.tesseract_dim, self.tesseract_dep, *args, **kwargs
        )
        self.xz_initializer = _TensorParallel2p5DXZGroupInitializer(
            self.tesseract_dim, self.tesseract_dep, *args, **kwargs
        )

    def init_dist_group(self):
        """
        Initialize 2.5D tensor parallel groups and assign local ranks and groups to each GPU.

        Returns:
            Dict: local_rank, group_world_size, process_group, ranks_in_group, mode
        """

        return [
            self.col_initializer.init_dist_group(),
            self.row_initializer.init_dist_group(),
            self.dep_initializer.init_dist_group(),
            self.xz_initializer.init_dist_group(),
        ]
