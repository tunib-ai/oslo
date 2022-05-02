import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import ColumnParallelLinear

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="1d",
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
input_ = torch.randn((4, 8)).cuda()
dist.broadcast(input_, src=0)

linear = torch.nn.Linear(8, 4).cuda()
if parallel_context.get_global_rank() == 0:
    out = linear(input_)
    print(f"original output: \n{out}\n")

dist.barrier()

# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = linear.weight.data.chunk(world_size, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
]

# split bias into 0:[0], 1:[1], 2:[2], 3:[3]
b = linear.bias.data.chunk(world_size, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
]

col_linear = ColumnParallelLinear(8, 4, parallel_context)
col_linear.weight.data = w
col_linear.bias.data = b

out = col_linear(input_)
out_list = [torch.zeros_like(out) for _ in range(world_size)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_1D))
out = torch.cat(out_list, dim=1)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
