import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear3D


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=8,
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
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out_update = linear(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

input_ = torch.chunk(input_, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
]
input_ = torch.chunk(input_, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]
input_ = torch.chunk(input_, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
]
target = torch.chunk(target, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
]
target = torch.chunk(target, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]
target = torch.chunk(target, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
]

w = torch.chunk(w, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
]
w = torch.chunk(w, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]
w = torch.chunk(w, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
]
b = torch.chunk(b, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]

linear_3d = Linear3D(input_dim, hidden_dim, parallel_context=parallel_context)
linear_3d.weight.data.copy_(w)
linear_3d.bias.data.copy_(b)

pout = linear_3d(input_)
optimizer = torch.optim.Adam(linear_3d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = linear_3d(input_)

pout_list = [torch.zeros_like(pout) for _ in range(cubic_dim)]
dist.all_gather(
    pout_list,
    pout.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_OUTPUT),
)
pout = torch.cat(pout_list, dim=-1)
pout_list = [torch.zeros_like(pout) for _ in range(cubic_dim)]
dist.all_gather(
    pout_list,
    pout.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_INPUT),
)
pout = torch.cat(pout_list, dim=0)
pout_list = [torch.zeros_like(pout) for _ in range(cubic_dim)]
dist.all_gather(
    pout_list,
    pout.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_WEIGHT),
)
pout = torch.cat(pout_list, dim=0)

pout_update_list = [torch.zeros_like(pout_update) for _ in range(cubic_dim)]
dist.all_gather(
    pout_update_list,
    pout_update.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_OUTPUT),
)
pout_update = torch.cat(pout_update_list, dim=-1)
pout_update_list = [torch.zeros_like(pout_update) for _ in range(cubic_dim)]
dist.all_gather(
    pout_update_list,
    pout_update.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_INPUT),
)
pout_update = torch.cat(pout_update_list, dim=0)
pout_update_list = [torch.zeros_like(pout_update) for _ in range(cubic_dim)]
dist.all_gather(
    pout_update_list,
    pout_update.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_WEIGHT),
)
pout_update = torch.cat(pout_update_list, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
