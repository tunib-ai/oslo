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
cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
input_ = torch.randn((4, 2)).cuda()
# target = torch.randn((4, 4)).cuda()
dist.broadcast(input_, src=0)
# dist.broadcast(target, src=0)

linear = torch.nn.Linear(2, 4).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)
dist.broadcast(w, src=0)
dist.broadcast(b, src=0)

out = linear(input_)
# optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
# logits = torch.nn.MSELoss()(out, target)
# logits.backward()
# optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{linear.weight.data}\n")
    print(f"original updated bias: \n{linear.bias.data}\n")

input_ = torch.chunk(input_, cubic_dim, dim=-2)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
]
input_ = torch.chunk(input_, cubic_dim, dim=-2)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]
input_ = torch.chunk(input_, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
]


w = torch.chunk(w, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_OUTPUT)
]
w = torch.chunk(w, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]
w = torch.chunk(w, cubic_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_WEIGHT)
]

b = torch.chunk(b, cubic_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_3D_INPUT)
]

linear_3d = Linear3D(2, 4, parallel_context=parallel_context)
linear_3d.weight.data.copy_(w)
# if parallel_context.get_global_rank() == 0:
#     import ipdb; ipdb.set_trace()
# dist.barrier()
linear_3d.bias.data.copy_(b)

out = linear_3d(input_)
# optimizer = torch.optim.Adam(linear_2d.parameters(), lr=1e-3)
# logits = torch.nn.MSELoss()(out, target)
# logits.backward()
# optimizer.step()

out_list = [torch.zeros_like(out) for _ in range(cubic_dim)]
dist.all_gather(
    out_list,
    out.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_INPUT),
)
out = torch.cat(out_list, dim=-2)
out_list = [torch.zeros_like(out) for _ in range(cubic_dim)]
dist.all_gather(
    out_list,
    out.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_OUTPUT),
)
out = torch.cat(out_list, dim=-1)
out_list = [torch.zeros_like(out) for _ in range(cubic_dim)]
dist.all_gather(
    out_list,
    out.contiguous(),
    parallel_context.get_group(ParallelMode.TENSOR_3D_WEIGHT),
)
out = torch.cat(out_list, dim=-2)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    # print(f"parallel updated weight: \n{w}\n")
    # print(f"original updated bias: \n{b}\n")
