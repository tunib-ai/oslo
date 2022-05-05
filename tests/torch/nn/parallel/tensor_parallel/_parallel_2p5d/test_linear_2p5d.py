import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2p5D

tp_size = 4
tp_depth = 4

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode="2.5d",
    tensor_parallel_depth=tp_depth
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.randn((4, 4)).cuda()
dist.broadcast(input_, src=0)

linear = torch.nn.Linear(4, 4).cuda()
if parallel_context.get_global_rank() == 0:
    out = linear(input_)
    print(f"original output: \n{out}\n")

dist.barrier()

# split input_ into
# 0:[0, 0, 0], 1:[0, 0, 1], 2:[0, 1, 0], 3:[0, 1, 1], 4:[1, 0, 0], 5:[1, 0, 1], 6:[1, 1, 0], 7:[1, 1, 1]
# input shape: (m/dq, n/q)
input_ = torch.chunk(input_, tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
]
input_ = torch.chunk(input_, tesseract_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
input_ = torch.chunk(input_, tp_depth, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
]

# split weight into 0,4:[0, 0], 1,5:[1, 0], 2,6:[0, 1], 3,7:[1, 1]
# input shape: (n/q, k/q)
w = linear.weight.data.chunk(tesseract_dim, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
w = w.chunk(tesseract_dim, dim=0)[parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)]

# split bias into 0,4:[0], 2,6:[1]
# input shape: (k/q)
b = linear.bias.data.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

linear_2p5d = Linear2p5D(4, 4, parallel_context)
linear_2p5d.weight.data = w
linear_2p5d.bias.data = b

out = linear_2p5d(input_)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW))
out = torch.cat(out_list, dim=1)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL))
out = torch.cat(out_list, dim=0)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_DEP))
out = torch.cat(out_list, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")