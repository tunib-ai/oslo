import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Embedding2p5D

tesseract_depth = 2
tp_size = 8

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode="2.5d",
    tensor_parallel_depth=tesseract_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
dist.broadcast(input_, src=0)

embedding = torch.nn.Embedding(10, 8).cuda()
if parallel_context.get_global_rank() == 0:
    out = embedding(input_)
    print(f"original output: \n{out}\n")
    o_out = out

dist.barrier()

# split weight into 0,4:[0], 1,5:[2], 2,6:[1], 3,7:[3]
w = embedding.weight.data.chunk(tesseract_dim, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
]
w = w.chunk(tesseract_dim, dim=1)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
]


embedding_2p5d = Embedding2p5D(10, 8, parallel_context)
embedding_2p5d.weight.data = w

out = embedding_2p5d(input_)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW))
out = torch.cat(out_list, dim=-1)
out_list = [torch.zeros_like(out) for _ in range(tesseract_dim)]
dist.all_gather(out_list, out, parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL))
out = torch.cat(out_list, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"difference: \n{(out - o_out)}\n")
