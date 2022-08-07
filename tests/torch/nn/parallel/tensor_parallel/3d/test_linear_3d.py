from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear3D
from _utils import split_input_3d, split_weight_3d, split_bias_3d, gather_output_3d

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
input_dim = 4
hidden_dim = 8
cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
input_ = torch.randn((batch_size, seq_len, input_dim)).cuda()
target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

linear = torch.nn.Linear(input_dim, hidden_dim).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)
orig_input_ = input_
orig_target = target

out = linear(input_)
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(out, target)
loss.backward()
optimizer.step()

out_update = linear(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

input_ = split_input_3d(input_, cubic_dim, parallel_context=parallel_context)
ptarget = split_input_3d(target, cubic_dim, parallel_context=parallel_context)

w = split_weight_3d(w, cubic_dim, parallel_context=parallel_context)
b = split_bias_3d(b, cubic_dim, parallel_context=parallel_context)

linear_3d = Linear3D(input_dim, hidden_dim, parallel_context=parallel_context)
linear_3d.weight.data.copy_(w)
linear_3d.bias.data.copy_(b)

pout = linear_3d(input_)
optimizer = torch.optim.Adam(linear_3d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, ptarget)
loss.backward()
optimizer.step()

pout_update = linear_3d(input_)

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

linear_3d = Linear3D(
    input_dim, hidden_dim, gather_output=True, parallel_context=parallel_context
)
linear_3d.weight.data.copy_(w)
linear_3d.bias.data.copy_(b)

pout = linear_3d(input_)
optimizer = torch.optim.Adam(linear_3d.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(pout, target)
loss.backward()
optimizer.step()

pout_update = linear_3d(input_)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output (gather_output=True): \n{pout}\n")
    print(f"parallel update output (gather_output=True): \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse (gather_output=True): \n{sse}\n")
    print(f"next output sse (gather_output=True): \n{sse_update}\n")
