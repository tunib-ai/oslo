import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import RowParallelLinear

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

# split input_ into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
input_ = input_.chunk(world_size, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
]

# split weight into 0:[0, 0], 1:[1, 0], 2:[0, 1], 3:[1, 1]
w = linear.weight.data.chunk(world_size, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
]

row_linear = RowParallelLinear(8, 4, parallel_context)
row_linear.weight.data = w
row_linear.bias.data = linear.bias.data

out = row_linear(input_)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
