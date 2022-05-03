import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import ColumnParallelLinear
from _utils import split_1d, gather_1d


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
target = torch.randn((4, 4)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

linear = torch.nn.Linear(8, 4).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)

out = linear(input_)
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{linear.weight.data}\n")
    print(f"original updated bias: \n{linear.bias.data}\n")

target = split_1d(parallel_context, target, world_size, dim=1)
# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = split_1d(parallel_context, w, world_size, dim=0)
# split bias into 0:[0], 1:[1], 2:[2], 3:[3]
b = split_1d(parallel_context, b, world_size, dim=0)

col_linear = ColumnParallelLinear(8, 4, parallel_context)
col_linear.weight.data = w
col_linear.bias.data = b

out = col_linear(input_)
optimizer = torch.optim.Adam(col_linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out = gather_1d(parallel_context, out, world_size, dim=1)
w = gather_1d(parallel_context, col_linear.weight.data, world_size, dim=0)
b = gather_1d(parallel_context, col_linear.bias.data, world_size, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
    print(f"original updated bias: \n{b}\n")
