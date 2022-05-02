import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding1D

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
dist.broadcast(input_, src=0)

vocab_embedding = torch.nn.Embedding(8, 10).cuda()
if parallel_context.get_global_rank() == 0:
    out = vocab_embedding(input_)
    print(f"original output: \n{out}\n")

dist.barrier()

# split weight into 0:[0], 1:[1], 2:[2], 3:[3]
w = vocab_embedding.weight.data.chunk(world_size, dim=0)[
    parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
]

vocab_embedding_1d = VocabParallelEmbedding1D(8, 10, parallel_context)
vocab_embedding_1d.weight.data = w

out = vocab_embedding_1d(input_)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
