import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed.parallel_mode import ParallelMode


class PipelineParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_group_size = self.world_size // self.data_parallel_size
        self.pipeline_stage_size = self.data_group_size // self.pipeline_parallel_size

    def init_dist_group(self):
        dist_settings = list()
        for i in range(self.data_parallel_size):
            for j in range(self.pipeline_stage_size):
                pipe_ranks = list(
                    range(
                        i * self.data_group_size + j,
                        (i + 1) * self.data_group_size,
                        self.pipeline_stage_size,
                    )
                )
                group_size = len(pipe_ranks)
                group = dist.new_group(pipe_ranks)
                group_cpu = (
                    dist.new_group(pipe_ranks, backend="gloo")
                    if dist.get_backend() != "gloo"
                    else group
                )

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    group_world_size = group_size
                    process_group = group
                    cpu_group = group_cpu
                    ranks_in_group = pipe_ranks
                    dist_settings.append(
                        {
                            "local_rank": local_rank,
                            "group_world_size": group_world_size,
                            "process_group": process_group,
                            "cpu_group": cpu_group,
                            "ranks_in_group": ranks_in_group,
                            "mode": ParallelMode.PIPELINE,
                        }
                    )

        return dist_settings
