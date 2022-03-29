from oslo.torch._context.initializers.initializer import ProcessGroupInitializer
from oslo.torch._context.parallel_mode import ParallelMode
import torch.distributed as dist


class DataParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_data_parallel_group = self.world_size // self.data_parallel_size

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.DATA

        for i in range(self.num_data_parallel_group):
            ranks = [
                i + j * self.num_data_parallel_group
                for j in range(self.data_parallel_size)
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
