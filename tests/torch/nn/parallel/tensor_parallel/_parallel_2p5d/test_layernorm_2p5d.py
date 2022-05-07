import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import LayerNorm2p5D

tp_size = 4
tesseract_depth = 1

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=tesseract_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.randn((4, 4)).cuda()
dist.broadcast(input_, src=0)

layernorm = torch.nn.LayerNorm(4).cuda()
if parallel_context.get_global_rank() == 0:
    out = layernorm(input_)
    print(f"original output: \n{out}\n")

dist.barrier()

# split input_ into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
input_ = torch.chunk(input_, tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
]
input_ = torch.chunk(input_, tesseract_dim, dim=-1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
input_ = torch.chunk(input_, tesseract_depth, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
]

# split weight into 0:[0], 1:[2], 2:[1], 3:[3]
w = layernorm.weight.data.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

# split bias into 0:[0], 1:[2], 2:[1], 3:[3]
b = layernorm.bias.data.chunk(tesseract_dim, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]

layernorm_2p5d = LayerNorm2p5D(4, parallel_context)
layernorm_2p5d.weight.data = w
layernorm_2p5d.bias.data = b

out = layernorm_2p5d(input_)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW))
out = torch.cat(out_list, dim=1)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL))
out = torch.cat(out_list, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
