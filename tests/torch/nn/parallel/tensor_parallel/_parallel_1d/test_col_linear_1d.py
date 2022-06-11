import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import ColLinear1D
from _utils import split_1d, gather_1d


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
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

out_update = linear(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original next output: \n{out_update}\n")

target = split_1d(parallel_context, target, world_size, dim=1)
# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = split_1d(parallel_context, w, world_size, dim=0)
# split bias into 0:[0], 1:[1], 2:[2], 3:[3]
b = split_1d(parallel_context, b, world_size, dim=0)

col_linear = ColLinear1D(8, 4, parallel_context=parallel_context)
col_linear.weight.data.copy_(w)
col_linear.bias.data.copy_(b)

pout = col_linear(input_)
optimizer = torch.optim.Adam(col_linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = col_linear(input_)

pout = gather_1d(parallel_context, pout, world_size, dim=1)
pout_update = gather_1d(parallel_context, pout_update, world_size, dim=1)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel next output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
