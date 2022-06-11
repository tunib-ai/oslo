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
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
input_ = torch.LongTensor([[0, 1, 6, 3, 8], [5, 2, 7, 4, 9]]).cuda()
target = torch.randn((2, 5, 8)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

vocab_embedding = torch.nn.Embedding(16, 8).cuda()
w = deepcopy(vocab_embedding.weight.data)

out = vocab_embedding(input_)
optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out_update = vocab_embedding(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original next output: \n{out_update}\n")

w = split_1d(w, world_size, dim=0, parallel_context=parallel_context)

vocab_embedding_1d = VocabParallelEmbedding1D(16, 8, parallel_context=parallel_context)
vocab_embedding_1d.weight.data = w

pout = vocab_embedding_1d(input_)
optimizer = torch.optim.Adam(vocab_embedding_1d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = vocab_embedding_1d(input_)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{pout}\n")
    print(f"parallel next output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
