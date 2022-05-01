import torch

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn import VocabParallelEmbedding2D


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2d",
)

ve_2d = VocabParallelEmbedding2D(20, 100, parallel_context)
input_ = torch.LongTensor([1,2,3,4,5,6,7,8])
out = ve_2d(input_.cuda())
print(out, out.shape)