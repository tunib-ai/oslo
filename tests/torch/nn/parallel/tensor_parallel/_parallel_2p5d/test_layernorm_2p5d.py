import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import LayerNorm2p5D

from _utils import *
from copy import deepcopy

tp_size = 8
tesseract_depth = 2

batch_size = 2
seq_len = 2
hidden_dim = 8

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=tesseract_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.randn((batch_size, seq_len, hidden_dim)).cuda()
target = torch.randn((batch_size, seq_len, hidden_dim)).cuda()

dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

layernorm = torch.nn.LayerNorm(hidden_dim).cuda()
w = deepcopy(layernorm.weight.data)
b = deepcopy(layernorm.bias.data)

out = layernorm(input_)
optimizer = torch.optim.Adam(layernorm.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out_update = layernorm(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

dist.barrier()

# split input_ into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
input_ = split_2d(parallel_context, input_, tesseract_dim, col_first=True)
target = split_2d(parallel_context, target, tesseract_dim, col_first=True)

# split weight into 0:[0], 1:[2], 2:[1], 3:[3]
w = w.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

# split bias into 0:[0], 1:[2], 2:[1], 3:[3]
b = b.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

layernorm_2p5d = LayerNorm2p5D(hidden_dim, parallel_context=parallel_context)
layernorm_2p5d.weight.data = w
layernorm_2p5d.bias.data = b

pout = layernorm_2p5d(input_)
optimizer = torch.optim.Adam(layernorm_2p5d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = layernorm_2p5d(input_)

pout = gather_2d(parallel_context, pout, tesseract_dim, col_first=False)
pout_update = gather_2d(parallel_context, pout_update, tesseract_dim, col_first=False)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
