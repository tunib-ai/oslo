from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import LayerNorm3D
from _utils import split_input_3d, split_layernorm_3d, split_bias_3d, gather_output_3d

tp_size = 8

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_3D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 4
seq_len = 2
hidden_dim = 8
cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
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

input_ = split_input_3d(input_, cubic_dim, parallel_context=parallel_context)
target = split_input_3d(target, cubic_dim, parallel_context=parallel_context)
w = split_layernorm_3d(w, cubic_dim, parallel_context=parallel_context)
b = split_bias_3d(b, cubic_dim, parallel_context=parallel_context)

layernorm_3d = LayerNorm3D(hidden_dim, parallel_context=parallel_context)
layernorm_3d.weight.data.copy_(w)
layernorm_3d.bias.data.copy_(b)

pout = layernorm_3d(input_)
optimizer = torch.optim.Adam(layernorm_3d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, target)
loss.backward()
optimizer.step()

pout_update = layernorm_3d(input_)

pout = gather_output_3d(pout, cubic_dim, parallel_context=parallel_context)
pout_update = gather_output_3d(
    pout_update, cubic_dim, parallel_context=parallel_context
)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
