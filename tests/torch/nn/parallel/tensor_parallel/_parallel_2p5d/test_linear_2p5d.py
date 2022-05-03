import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2p5D

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2.5d",
    tensor_parallel_depth=1
)

# col_parallel_mode = ParallelMode.TENSOR_2P5D_COL
# print(parallel_context.get_group(col_parallel_mode).rank())
#
# row_parallel_mode = ParallelMode.TENSOR_2P5D_ROW
# print(parallel_context.get_group(row_parallel_mode).rank())

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
input_ = torch.chunk(input_, 2, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
]
input_ = torch.chunk(input_, 2, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
# input_ = torch.chunk(input_, 1, dim=0)[
#     parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
# ]

# split weight into 0:[0, 0], 1:[1, 0], 2:[0, 1], 3:[1, 1]
# input shape: (n/q, k/q)
w = linear.weight.data.chunk(2, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
w = w.chunk(2, dim=0)[parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)]

# split bias into 0:[0], 1:[2], 2:[1], 3:[3]
b = linear.bias.data.chunk(2, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
b = b.chunk(2, dim=0)[parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)]

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
# out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
# dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_DEP))
# out = torch.cat(out_list, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
