import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed.parallel_mode import ParallelMode


class _SequenceParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_sequence_parallel_group = (
            self.world_size // self.sequence_parallel_size
        )

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.SEQUENCE

        for i in range(self.num_sequence_parallel_group):
            ranks = [
                i * self.sequence_parallel_size + j
                for j in range(self.sequence_parallel_size)
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


class _SequenceDataParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_size = self.world_size // self.pipeline_parallel_size
        self.num_group = self.pipeline_parallel_size

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        cpu_group = None
        group_world_size = None
        mode = ParallelMode.SEQUENCE_DP

        for i in range(self.num_group):
            ranks = [i * self.dp_size + j for j in range(self.dp_size)]
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


class SequenceParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sequence_initializer = _SequenceParallelGroupInitializer(*args, **kwargs)
        self._sequence_dp_initializer = _SequenceDataParallelGroupInitializer(
            *args, **kwargs
        )

    def init_dist_group(self):
        return [
            self._sequence_initializer.init_dist_group(),
            self._sequence_dp_initializer.init_dist_group(),
        ]
