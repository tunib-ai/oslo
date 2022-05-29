import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Embedding2D
from _utils import split_2d, split_1d_twice, gather_2d, gather_1d_twice


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode=ParallelMode.TENSOR_2D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
input_ = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
target = torch.randn((2, 4, 8)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

embedding = torch.nn.Embedding(10, 8).cuda()
w = deepcopy(embedding.weight.data)

out = embedding(input_)
optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{embedding.weight.data}\n")

# split target into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
target = split_2d(parallel_context, target, summa_dim, col_first=True)
# split weight into 0:[0], 1:[2], 2:[1], 3:[3]
w = split_1d_twice(parallel_context, w, summa_dim, dim=1)

embedding_2d = Embedding2D(10, 8, parallel_context=parallel_context)
embedding_2d.weight.data = w

out = embedding_2d(input_)
optimizer = torch.optim.Adam(embedding_2d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out = gather_2d(parallel_context, out, summa_dim, col_first=False)
w = gather_1d_twice(parallel_context, embedding_2d.weight.data, summa_dim, dim=1)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
