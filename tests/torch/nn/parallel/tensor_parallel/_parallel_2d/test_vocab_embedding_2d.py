import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding2D
from utils import split_2d, gather_2d


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2d",
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
input_ = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
target = torch.randn((2, 4, 8)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

vocab_embedding = torch.nn.Embedding(10, 8).cuda()
w = deepcopy(vocab_embedding.weight.data)

out = vocab_embedding(input_)
optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{vocab_embedding.weight.data}\n")

# split target into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
target = split_2d(parallel_context, target, summa_dim, col_first=True)
# split weight into 0:[0, 0], 1:[1, 0], 2:[0, 1], 3:[1, 1]
w = split_2d(parallel_context, w, summa_dim, col_first=False)

vocab_embedding_2d = VocabParallelEmbedding2D(10, 8, parallel_context)
vocab_embedding_2d.weight.data = w

out = vocab_embedding_2d(input_)
optimizer = torch.optim.Adam(vocab_embedding_2d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out = gather_2d(parallel_context, out, summa_dim, col_first=False)
w = gather_2d(parallel_context, vocab_embedding_2d.weight.data, summa_dim, col_first=True)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
