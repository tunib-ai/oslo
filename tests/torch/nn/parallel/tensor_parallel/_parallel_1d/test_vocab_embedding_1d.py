import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding1D
from _utils import split_1d, gather_1d


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="1d",
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
input_ = torch.LongTensor([[1, 2], [5, 6]]).cuda()
target = torch.randn((2, 2, 10)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

vocab_embedding = torch.nn.Embedding(8, 10).cuda()
w = deepcopy(vocab_embedding.weight.data)

out = vocab_embedding(input_)
optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{vocab_embedding.weight.data}\n")

# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = split_1d(parallel_context, w, world_size, dim=0)

vocab_embedding_1d = VocabParallelEmbedding1D(8, 10, parallel_context)
vocab_embedding_1d.weight.data = w

out = vocab_embedding_1d(input_)
optimizer = torch.optim.Adam(vocab_embedding_1d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

w = gather_1d(parallel_context, w, world_size, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
