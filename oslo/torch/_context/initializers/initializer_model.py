from oslo.torch._context.initializers.initializer import ProcessGroupInitializer
from oslo.torch._context.parallel_mode import ParallelMode
import torch.distributed as dist


class ModelParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_parallel_size = (
            self.tensor_parallel_size * self.pipeline_parallel_size
        )
        self.num_group = self.world_size // self.model_parallel_size

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.MODEL

        for i in range(self.num_group):
            ranks = [
                i * self.model_parallel_size + j
                for j in range(self.model_parallel_size)
            ]

            group = dist.new_group(ranks)

            if self.rank in ranks:
                local_rank = ranks.index(self.rank)
                group_world_size = len(ranks)
                process_group = group
                ranks_in_group = ranks

        return {
            "local_rank": local_rank,
            "group_world_size": group_world_size,
            "process_group": process_group,
            "ranks_in_group": ranks_in_group,
            "mode": mode,
        }
