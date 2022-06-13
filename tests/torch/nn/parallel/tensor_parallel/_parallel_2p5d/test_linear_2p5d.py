import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2p5D

from _utils import split_2p5d, split_2d, gather_2p5d, gather_2d

from copy import deepcopy

tp_size = 8
tp_depth = 2

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=tp_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.randn((4, 3, 4)).cuda()
target = torch.randn((4, 3, 4)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

linear = torch.nn.Linear(4, 4).cuda()
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
    print(f"original updated weight: \n{linear.weight.data}\n")
    print(f"original updated bias: \n{linear.bias.data}\n")

# split input_ into
# 0:[0, 0, 0], 1:[0, 0, 1], 2:[0, 1, 0], 3:[0, 1, 1], 4:[1, 0, 0], 5:[1, 0, 1], 6:[1, 1, 0], 7:[1, 1, 1]
# input shape: (m/dq, n/q)
input_ = split_2d(parallel_context, input_, tesseract_dim)
ptarget = split_2d(parallel_context, target, tesseract_dim)

# split weight into 0,4:[0, 0], 1,5:[1, 0], 2,6:[0, 1], 3,7:[1, 1]
# input shape: (n/q, k/q)
w = split_2d(parallel_context, w, tesseract_dim, col_first=False)
# split bias into 0,4:[0], 2,6:[1]
# input shape: (k/q)
b = b.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

linear_2p5d = Linear2p5D(4, 4, parallel_context=parallel_context)
linear_2p5d.weight.data.copy_(w)
linear_2p5d.bias.data.copy_(b)

pout = linear_2p5d(input_)
optimizer = torch.optim.Adam(linear_2p5d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, ptarget)
logits.backward()
optimizer.step()

pout_update = linear_2p5d(input_)

pout = gather_2d(parallel_context, pout, tesseract_dim, False)
pout_update = gather_2d(parallel_context, pout_update, tesseract_dim, False)

# w = gather_2d(parallel_context, linear_2p5d.weight.data, tesseract_dim, True)
# b = gather_1d(parallel_context, linear_2p5d.bias.data, tesseract_dim, 0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    minmax_update = (out_update - pout_update) ** 2
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
    print(f"next output max: \n{minmax_update.max()}\n")
    print(f"next output min: \n{minmax_update.min()}\n")




linear_2p5d = Linear2p5D(4, 4, gather_output=True, parallel_context=parallel_context)
linear_2p5d.weight.data.copy_(w)
linear_2p5d.bias.data.copy_(b)

pout = linear_2p5d(input_)
optimizer = torch.optim.Adam(linear_2p5d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = linear_2p5d(input_)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output (gather_output=True): \n{pout}\n")
    print(f"parallel update output (gather_output=True): \n{pout_update}\n")


if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    minmax_update = (out_update - pout_update) ** 2
    print(f"output sse (gather_output=True): \n{sse}\n")
    print(f"next output sse (gather_output=True): \n{sse_update}\n")
    import pprint
    # top5 = torch.clamp(minmax_update.flatten(), 1e-8)
    top5 = minmax_update.flatten()
    top5 = [t.item() for t in top5]
    top5 = [top5[i:i+4] for i in range(0, len(top5), 4)]
    pprint.pprint(top5)
    print(f"next output min: \n{minmax_update.min()}\n")
