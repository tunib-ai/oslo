import argparse
from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import LayerNorm1D
from _utils import split_1d, gather_1d


parser = argparse.ArgumentParser()
parser.add_argument("--memory_priority", action="store_true", default=False)
args = parser.parse_args()

tp_size = 4

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    memory_priority=args.memory_priority,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 2
seq_len = 4
hidden_dim = 8
world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
input_ = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()

dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

layernorm = torch.nn.LayerNorm(hidden_dim).cuda()
w = deepcopy(layernorm.weight.data)
b = deepcopy(layernorm.bias.data)

out = layernorm(input_)
optimizer = torch.optim.Adam(layernorm.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(out, target)
loss.backward()
optimizer.step()

out_update = layernorm(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

if args.memory_priority:
    input_ = split_1d(input_, world_size, dim=1, parallel_context=parallel_context)
    target = split_1d(target, world_size, dim=1, parallel_context=parallel_context)

layernorm_1d = LayerNorm1D(hidden_dim, parallel_context=parallel_context)
layernorm_1d.weight.data = w
layernorm_1d.bias.data = b

pout = layernorm_1d(input_)
optimizer = torch.optim.Adam(layernorm_1d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, target)
loss.backward()
optimizer.step()

pout_update = layernorm_1d(input_)
if args.memory_priority:
    pout = gather_1d(pout, world_size, dim=1, parallel_context=parallel_context)
    pout_update = gather_1d(pout_update, world_size, dim=1, parallel_context=parallel_context)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
