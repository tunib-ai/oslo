import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params


torch.manual_seed(42)

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
)

current_device = torch.cuda.current_device()

n_steps = 20
batch_size = 16

in_channels = 16
hidden_channels = 8
out_channels = 4

fc1 = nn.Linear(in_channels, hidden_channels)
fc2 = nn.Linear(hidden_channels, out_channels)
model = nn.Sequential(fc1, fc2)

fc1_no_pp = nn.Linear(in_channels, hidden_channels)
fc2_no_pp = nn.Linear(hidden_channels, out_channels)
model_no_pp = nn.Sequential(fc1_no_pp, fc2_no_pp)
model_no_pp.load_state_dict(model.state_dict())
model_no_pp.to(current_device)

wrapper_pp = PipelineParallel(
    model, parallel_context=parallel_context, micro_batch_size=4, use_auto_partitioning=True, memory_computation_balance=1.0
)

wrapper_pp.train()

if parallel_context.get_global_rank() == 0:
    print(wrapper_pp.partitioner.module)

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-5)
optimizer_no_pp = Adam(model_no_pp.parameters(), lr=3e-5)

allocate_params(wrapper_pp, parallel_context)
"""
for rank in range(dist.get_world_size()):
    if dist.get_rank() == rank:
        print(f"RANK: {rank}:")
        num_params = 0
        for name, param in wrapper_pp.named_parameters():
            if param.device != torch.device("cpu"):
                print(f"> {name}: {param.device}")
                num_params += param.numel()
            else:
                print(f"> {name}: {param.device}")
        print(f"RANK {rank} params: {num_params}")
    dist.barrier()
    print()
"""

loss_fn = torch.nn.MSELoss()

for i in range(n_steps):
    sample_input = torch.rand(batch_size, in_channels).to(current_device)
    sample_output = torch.rand(batch_size, out_channels)

    optimizer_pp.zero_grad()
    optimizer_no_pp.zero_grad()

    out_pp = wrapper_pp(sample_input)
    out_no_pp = model_no_pp(sample_input)
    sample_output = sample_output.to(out_pp.device)
    loss_pp = loss_fn(out_pp, sample_output)
    loss_no_pp = loss_fn(out_no_pp, sample_output)

    if parallel_context.get_global_rank() == 0:
        print(f"rank: {dist.get_rank()}, pp:{loss_pp}, NOTHING:{loss_no_pp}")

    loss_pp.backward()
    loss_no_pp.backward()

    optimizer_pp.step()
    optimizer_no_pp.step()
