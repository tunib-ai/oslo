from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import ColLinear1D
from _utils import split_1d, gather_1d

tp_size = 4

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 2
seq_len = 2
input_dim = 4
hidden_dim = 8
world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
input_ = torch.randn((batch_size, seq_len, input_dim)).cuda()
target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

linear = torch.nn.Linear(input_dim, hidden_dim).cuda()
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

target = split_1d(target, world_size, dim=-1, parallel_context=parallel_context)
w = split_1d(w, world_size, dim=0, parallel_context=parallel_context)
b = split_1d(b, world_size, dim=0, parallel_context=parallel_context)

col_linear = ColLinear1D(input_dim, hidden_dim, parallel_context=parallel_context)
col_linear.weight.data.copy_(w)
col_linear.bias.data.copy_(b)

pout = col_linear(input_)
optimizer = torch.optim.Adam(col_linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = col_linear(input_)

pout = gather_1d(pout, world_size, dim=-1, parallel_context=parallel_context)
pout_update = gather_1d(
    pout_update, world_size, dim=-1, parallel_context=parallel_context
)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel next output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
