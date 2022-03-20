from oslo.torch._context.initializers.initializer import ProcessGroupInitializer
import torch.distributed as dist

from oslo.torch._context.parallel_mode import ParallelMode


class TensorParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tensor_parallel_group = self.world_size // self.tensor_parallel_size

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.TENSOR

        for i in range(self.num_tensor_parallel_group):
            ranks = [
                i * self.tensor_parallel_size + j
                for j in range(self.tensor_parallel_size)
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
