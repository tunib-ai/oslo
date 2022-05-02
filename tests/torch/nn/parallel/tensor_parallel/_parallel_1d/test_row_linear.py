import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import RowParallelLinear
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

# split input_ into 0:[0], 1:[1], 2:[2], 3:[3]
input_ = split_1d(parallel_context, input_, world_size, dim=1)
# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = split_1d(parallel_context, w, world_size, dim=1)

row_linear = RowParallelLinear(8, 4, parallel_context)
row_linear.weight.data = w
row_linear.bias.data = b

out = row_linear(input_)
optimizer = torch.optim.Adam(row_linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

w = gather_1d(parallel_context, w, world_size, dim=1)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
    print(f"original updated bias: \n{b}\n")
