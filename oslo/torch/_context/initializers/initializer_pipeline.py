from oslo.torch._context.initializers.initializer import ProcessGroupInitializer
import torch.distributed as dist

from oslo.torch._context.parallel_mode import ParallelMode


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
                pipe_group_size = len(pipe_ranks)
                pipe_group = dist.new_group(pipe_ranks)

                if self.rank in pipe_ranks:
                    local_rank = pipe_ranks.index(self.rank)
                    group_world_size = pipe_group_size
                    process_group = pipe_group
                    ranks_in_group = pipe_ranks
                    dist_settings.append(
                        {
                            "local_rank": local_rank,
                            "group_world_size": group_world_size,
                            "process_group": process_group,
                            "ranks_in_group": ranks_in_group,
                            "mode": ParallelMode.PIPELINE,
                        }
                    )

        return dist_settings
