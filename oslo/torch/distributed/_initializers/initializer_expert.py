import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer import (
    ProcessGroupInitializer,
)
from oslo.torch.distributed.parallel_mode import ParallelMode


# TODO : Need to Test
class ExpertParallelGroupInitializer(ProcessGroupInitializer):

    BASE_ELEMENT = [
        "local_rank",
        "group_world_size",
        "process_group",
        "cpu_group",
        "ranks_in_group",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dp_world_size = self.world_size // self.tensor_parallel_size
        self.dp_group_size = self.tensor_parallel_size

        # if isinstance(self.expert_parallel_size, dict):
        #    self.num_expert_parallel_group = {
        #        k: self.world_size // v
        #        for k, v in self.expert_parallel_size.items()
        #    }
        # elif isinstance(self.expert_parallel_size, int):
        #    self.num_expert_parallel_group = self.world_size // self.expert_parallel_size
        # else:
        #    raise TypeError("Type {type(self.expert_parallel_size) is not supported now for expert parallel size}.")

    def init_dist_group(self):
        if isinstance(self.expert_parallel_size, dict):
            # 5 = len([local_rank, group_world_size, process_group, cpu_group, ranks_in_group])
            ep_info = [
                {"enc": dict(), "dec": dict()}
                for i in range(len(ExpertParallelGroupInitializer.BASE_ELEMENT))
            ]
            # edp_info = [{"enc": dict(), "dec": dict()} for i in range(5)]

            # Encoder
            if "enc" in self.expert_parallel_size:
                # self.construct_stack_parallel_info(ep_info, edp_info, stack="enc")
                self.construct_stack_parallel_info(ep_info, stack="enc")

            # Decoder
            if "dec" in self.expert_parallel_size:
                # self.construct_stack_parallel_info(ep_info, edp_info, stack="dec")
                self.construct_stack_parallel_info(ep_info, stack="dec")

            ep_info += [ParallelMode.EXPERT]
        elif isinstance(self.expert_parallel_size, int):
            # ep_info, edp_info = self.init_dist_group_per_unit(self.expert_parallel_size)
            ep_info = self.init_dist_group_per_unit(self.expert_parallel_size)
            ep_info = {
                k: v
                for k, v in zip(ExpertParallelGroupInitializer.BASE_ELEMENT, ep_info)
            }
            ep_info["mode"] = ParallelMode.EXPERT

        # ep_info = [
        #       local_rank_dict, group_world_size_dict,
        #       process_group_dict, cpu_group_dict, ranks_in_group_dict,
        #       mode
        # ]
        # edp_info += [ParallelMode.EXPERT_DATA]

        # return [ep_info, edp_info]
        return ep_info

    # def construct_stack_parallel_info(self, ep_info, edp_info, stack=None):
    def construct_stack_parallel_info(self, ep_info, stack=None):

        for k, v in self.expert_parallel_size[stack].items():
            # cur_ep_info, cur_edp_info = self.init_dist_group_per_unit(v)
            cur_ep_info = self.init_dist_group_per_unit(v)

            # for i, info in enumerate(zip(cur_ep_info, cur_edp_info)):
            for i, info in enumerate(cur_ep_info):
                # ep_info[i][stack][k], edp_info[i][stack][k] = info
                ep_info[i][stack][k] = info

        return

    def init_dist_group_per_unit(self, expert_parallel_size):
        # Construct Data Parallel Group
        data_parallel_groups = list()
        for i in range(self.dp_group_size):
            data_parallel_groups.append(
                list(range(i, self.world_size, self.dp_group_size))
            )

        # expert_parallel_info, expert_data_parallel_info = None, None
        expert_parallel_info = None
        for dp_ranks in data_parallel_groups:
            # part_ep_groups = list()

            # Construct Expert Parallel Group
            for i in range(0, self.dp_world_size, expert_parallel_size):
                ranks = dp_ranks[i : i + expert_parallel_size]
                # part_ep_groups.append(ranks)
                process_group = dist.new_group(ranks)
                group_cpu = (
                    dist.new_group(ranks, backend="gloo")
                    if dist.get_backend() != "gloo"
                    else process_group
                )

                if self.rank in ranks:
                    expert_parallel_info = self.construct_parallel_info(
                        ranks, process_group, group_cpu
                    )

            # Construct Expert Data Parallel Group
            # for expert_dp_ranks in zip(*part_ep_groups):
            #    ranks = list(expert_dp_ranks)
            #    if self.rank in ranks:
            #        expert_data_parallel_info = self.construct_parallel_info(ranks)

        # return expert_parallel_info, expert_data_parallel_info
        return expert_parallel_info

    def construct_parallel_info(self, ranks, process_group, cpu_group):
        local_rank = ranks.index(self.rank)
        group_world_size = len(ranks)
        # process_group = dist.new_group(ranks)
        # cpu_group = (
        #    dist.new_group(ranks, backend="gloo")
        #    if dist.get_backend() != "gloo"
        #    else process_group
        # )
        ranks_in_group = ranks

        return [
            local_rank,
            group_world_size,
            process_group,
            cpu_group,
            ranks_in_group,
        ]
