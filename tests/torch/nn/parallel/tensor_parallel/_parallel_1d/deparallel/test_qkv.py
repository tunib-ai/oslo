import torch.nn as nn

from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._wrapper import _TensorParallel1D
from oslo.torch.nn import ColumnParallelLinear, RowParallelLinear
from oslo.torch.distributed import ParallelContext, ParallelMode
from copy import deepcopy
from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import gather_1d

tp_size = 4
tp_depth = 2

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
    tensor_parallel_depth=tp_depth,
)

world_size = parallel_context.get_world_size(ParallelMode.TENSOR_1D)
rank = parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
fusion_degree = 3

linear = nn.Linear(4, fusion_degree*4).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)

dim= 1

weight_list = w.t().chunk(fusion_degree * world_size, dim=dim)
bias_list = b.chunk(fusion_degree * world_size, dim=0)

# [t][f*t]
weight_list = _TensorParallel1D._deconstruct_combined_qkv(weight_list, world_size, fusion_degree, dim)
chunked_w = weight_list[rank].contiguous()
bias_list = _TensorParallel1D._deconstruct_combined_qkv(bias_list, world_size, fusion_degree, 0)
chunked_b = bias_list[rank].contiguous()

linear_1d = RowParallelLinear(4, fusion_degree * 4, parallel_context=parallel_context, bias=True)
if parallel_context.get_global_rank() == 0:
    print(chunked_w.size())
    print(linear_1d.weight.data.size())
linear_1d.weight.data = chunked_w

recon_chunked_w = gather_1d(parallel_context, linear_1d.weight.data, world_size, dim)
recon_chunked_b = gather_1d(parallel_context, linear_1d.bias.data, world_size, 0)
# reshaped_w = recon_chunked_w.view(-1, tesseract_dim, 4)
# recon_w = torch.cat([reshaped_w[:3], reshaped_w[3:]], 1).view(-1, 4).contiguous()
print(recon_chunked_w.shape)
recon_w = _TensorParallel1D._reconstruct_combined_qkv(recon_chunked_w, world_size, fusion_degree, dim)
recon_b = _TensorParallel1D._reconstruct_combined_qkv(recon_chunked_b, world_size, fusion_degree, 0)

if parallel_context.get_global_rank() == 0:
    print(f"original w: \n{w}\n")
    print(f"reconstruct w: \n{recon_w}\n")

    print(f"original b: \n{b}\n")
    print(f"reconstruct b: \n{recon_b}\n")