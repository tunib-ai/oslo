from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import LayerNorm2D
from _utils import split_2d, split_layernorm_2d, split_bias_2d, gather_2d

tp_size = 4

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 2
seq_len = 2
hidden_dim = 8
summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
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

dist.barrier()

input_ = split_2d(input_, summa_dim, parallel_context=parallel_context)
target = split_2d(target, summa_dim, parallel_context=parallel_context)
w = split_layernorm_2d(w, summa_dim, parallel_context=parallel_context)
b = split_bias_2d(b, summa_dim, parallel_context=parallel_context)

layernorm_2d = LayerNorm2D(hidden_dim, parallel_context=parallel_context)
layernorm_2d.weight.data.copy_(w)
layernorm_2d.bias.data.copy_(b)

pout = layernorm_2d(input_)
optimizer = torch.optim.Adam(layernorm_2d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, target)
loss.backward()
optimizer.step()

pout_update = layernorm_2d(input_)

pout = gather_2d(pout, summa_dim, parallel_context=parallel_context)
pout_update = gather_2d(pout_update, summa_dim, parallel_context=parallel_context)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
