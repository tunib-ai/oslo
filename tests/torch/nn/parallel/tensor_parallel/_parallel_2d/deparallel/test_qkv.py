import torch.nn as nn

from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._wrapper import (
    _TensorParallel2D,
)
from oslo.torch.nn import Linear2D
from oslo.torch.distributed import ParallelContext, ParallelMode
from copy import deepcopy
from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
    gather_1d,
    gather_2d,
)

tp_size = 4
tp_depth = 1

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2D,
    tensor_parallel_depth=tp_depth,
)

row_rank = parallel_context.get_local_rank(ParallelMode.TENSOR_2D_ROW)
col_rank = parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
fusion_degree = 3

linear = nn.Linear(4, fusion_degree * 4).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)

weight_list = w.chunk(summa_dim, dim=1)
weight_list = [weight.chunk(summa_dim * fusion_degree, dim=0) for weight in weight_list]
bias_list = b.chunk(summa_dim * fusion_degree, dim=0)

# [t][f*t]
weight_list = _TensorParallel2D._deconstruct_combined_qkv(
    weight_list, summa_dim, fusion_degree
)
bias_list = _TensorParallel2D._deconstruct_combined_qkv(
    bias_list, summa_dim, fusion_degree
)
chunked_w = weight_list[row_rank][col_rank]
chunked_b = bias_list[row_rank]

linear_2d = Linear2D(4, fusion_degree * 4, parallel_context=parallel_context, bias=True)
if parallel_context.get_global_rank() == 0:
    print(chunked_w.size())
    print(linear_2d.weight.data.size())
linear_2d.weight.data = chunked_w
linear_2d.bias.data = chunked_b

recon_chunked_w = gather_2d(parallel_context, linear_2d.weight.data, summa_dim, True)
recon_chunked_b = gather_1d(parallel_context, linear_2d.bias.data, summa_dim, 0)
# reshaped_w = recon_chunked_w.view(-1, tesseract_dim, 4)
# recon_w = torch.cat([reshaped_w[:3], reshaped_w[3:]], 1).view(-1, 4).contiguous()

recon_w = _TensorParallel2D._reconstruct_combined_qkv(
    recon_chunked_w, summa_dim, fusion_degree, False
)
recon_b = _TensorParallel2D._reconstruct_combined_qkv(
    recon_chunked_b, summa_dim, fusion_degree, True
)

if parallel_context.get_global_rank() == 0:
    print(f"original w: \n{w}\n")
    print(f"reconstruct w: \n{recon_w}\n")

    print(f"original b: \n{b}\n")
    print(f"reconstruct b: \n{recon_b}\n")
