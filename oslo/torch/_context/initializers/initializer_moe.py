from oslo.torch._context.initializers.initializer import ProcessGroupInitializer
from oslo.torch._context.parallel_mode import ParallelMode
import torch.distributed as dist


class _MoEModelParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, moe_model, moe_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
        self.moe_data = moe_data

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.MOE_MODEL

        for i in range(self.moe_data):
            ranks = [i * self.moe_model + j for j in range(self.moe_model)]
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


class _MoEDataParallelGroupInitializer(ProcessGroupInitializer):
    def __init__(self, moe_model, moe_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
        self.moe_data = moe_data

    def init_dist_group(self):
        local_rank = None
        ranks_in_group = None
        process_group = None
        group_world_size = None
        mode = ParallelMode.MOE_DATA

        for i in range(self.moe_model):
            ranks = [i + j * self.moe_model for j in range(self.moe_data)]
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


class MoEGroupInitializer(ProcessGroupInitializer):
    def __init__(self, moe_model, moe_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_model = moe_model
        self.moe_data = moe_data

        self.model_initializer = _MoEModelParallelGroupInitializer(
            self.moe_model, self.moe_data, *args, **kwargs
        )
        self.data_initializer = _MoEDataParallelGroupInitializer(
            self.moe_model, self.moe_data, *args, **kwargs
        )

    def init_dist_group(self):
        return [
            self.model_initializer.init_dist_group(),
            self.data_initializer.init_dist_group(),
        ]
