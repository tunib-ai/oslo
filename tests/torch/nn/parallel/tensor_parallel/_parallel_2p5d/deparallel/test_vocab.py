import torch
import torch.distributed as dist

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import VocabParallelEmbedding2p5D
from oslo.torch.nn.parallel import utils

from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import split_batch_2d, split_2d, gather_2d

from copy import deepcopy


tp_size = 8
tp_depth = 2

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=tp_depth,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
tesseract_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
input_ = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).cuda()
target = torch.randn((2, 4, 16)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

vocab_embedding = torch.nn.Embedding(10, 16).cuda()
w = deepcopy(vocab_embedding.weight.data)

out = vocab_embedding(input_)
optimizer = torch.optim.Adam(vocab_embedding.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out_update = vocab_embedding(input_)

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original update output: \n{out_update}\n")
    # print(f"vocab start: {vocab_embedding.start_index}, vocab end: {vocab_embedding.end_index}")

input_ = split_batch_2d(parallel_context, input_, tesseract_dim)
# split target into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
target = split_2d(parallel_context, target, tesseract_dim, col_first=True)
# split weight into 0:[0, 0], 1:[1, 0], 2:[0, 1], 3:[1, 1]
w = split_2d(parallel_context, w, tesseract_dim, col_first=False)

vocab_embedding_2p5d = VocabParallelEmbedding2p5D(
    10, 16, parallel_context=parallel_context
)
vocab_embedding_2p5d.weight.data.copy_(w)

pout = vocab_embedding_2p5d(input_)
optimizer = torch.optim.Adam(vocab_embedding_2p5d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(pout, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    unwrapped_model = utils.unwrap_parallel(vocab_embedding_2p5d)
    print(f"original vocab size: {unwrapped_model.orig_vocab_size}")


#
#
# pout_update = vocab_embedding_2p5d(input_)
#
# pout = gather_2d(parallel_context, pout, tesseract_dim, col_first=False)
# pout_update = gather_2d(parallel_context, pout_update, tesseract_dim, col_first=False)
#
# if parallel_context.get_global_rank() == 0:
#     print(f"parallel output: \n{pout}\n")
#     print(f"parallel update output: \n{pout_update}\n")
#
# if parallel_context.get_global_rank() == 0:
#     sse = torch.sum((out - pout) ** 2).item()
#     sse_update = torch.sum((out_update - pout_update) ** 2).item()
#     print(f"output sse: \n{sse}\n")
#     print(f"next output sse: \n{sse_update}\n")