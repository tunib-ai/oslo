from copy import deepcopy
import torch
import torch.distributed as dist
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding3D
from _utils import split_batch_3d, split_input_3d, split_weight_3d, gather_output_3d

tp_size = 8

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_3D,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

batch_size = 4
seq_len = 5
num_embeddings = 16
embedding_dim = 8
cubic_dim = parallel_context.get_world_size(ParallelMode.TENSOR_3D_INPUT)
input_ = torch.LongTensor([
    [0, 1, 6, 13, 8], 
    [5, 12, 7, 4, 9], 
    [5, 2, 7, 15, 4], 
    [14, 2, 8, 7, 9]]).cuda()
target = torch.randn((batch_size, seq_len, embedding_dim)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

embedding = torch.nn.Embedding(num_embeddings, embedding_dim).cuda()
w = deepcopy(embedding.weight.data)

out = embedding(input_)
optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out_update = embedding(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")

input_ = split_batch_3d(input_, cubic_dim, parallel_context=parallel_context)
target = split_input_3d(target, cubic_dim, parallel_context=parallel_context)
w = split_weight_3d(w, cubic_dim, parallel_context=parallel_context)

vocab_embedding_3d = VocabParallelEmbedding3D(num_embeddings, embedding_dim, parallel_context=parallel_context)
vocab_embedding_3d.weight.data.copy_(w)

pout = vocab_embedding_3d(input_)
optimizer = torch.optim.Adam(vocab_embedding_3d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

pout_update = vocab_embedding_3d(input_)

pout = gather_output_3d(pout, cubic_dim, parallel_context=parallel_context)
pout_update = gather_output_3d(pout_update, cubic_dim, parallel_context=parallel_context)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel update output: \n{pout_update}\n")

if parallel_context.get_global_rank() == 0:
    sse = torch.sum((out - pout) ** 2).item()
    sse_update = torch.sum((out_update - pout_update) ** 2).item()
    print(f"output sse: \n{sse}\n")
    print(f"next output sse: \n{sse_update}\n")
