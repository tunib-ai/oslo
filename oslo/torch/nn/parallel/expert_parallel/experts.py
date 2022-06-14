import torch
import copy


class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.experts = torch.nn.Module(
            [copy.deepcopy(expert) for i in range(num_local_experts)]
        )

        self.num_local_experts = num_local_experts

        for expert in self.experts:
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        # |inputs| = (ep_size, num_local_experts, capacity, d_model)
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        # |chunks| = (num_local_experts, )
        #   |chunks[i]| = (ep_size, 1, capacity, d_model)

        expert_outputs = list()
        for chunk, expert in zip(chunks, self.experts):
            out = expert(chunk)
            # |out|
            # = (ep_size, 1, capacity, 4*d_model), for ExpertParallelFrontBlock
            # = (ep_size, 1, capacity, d_model) for ExpertParallelBehindBlock

            # ignore the bias term for now
            if type(out) is tuple:
                out = out[0]

            expert_outputs += [out]
        expert_output = torch.cat(expert_outputs, dim=1)
        # |expert_output|
        # = (ep_size, num_local_experts, capacity, 4*d_model), for ExpertParallelFrontBlock
        # = (ep_size, num_local_experts, capacity, d_model) for ExpertParallelBehindBlock
        return expert_output
