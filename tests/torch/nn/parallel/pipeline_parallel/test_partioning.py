import torch
import torch.distributed as dist
from transformers import GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params

parallel_context = ParallelContext.from_torch(pipeline_parallel_size=4)
model = GPT2LMHeadModel.from_pretrained("gpt2")

wrapper_pp = PipelineParallel(
    model, parallel_context=parallel_context, memory_computation_balance=0.5, micro_batch_size=4
)
allocate_params(wrapper_pp, parallel_context)

for rank in range(dist.get_world_size()):
    if dist.get_rank() == rank:
        print(f"RANK: {rank}:")
        num_params = 0
        for name, param in wrapper_pp.named_parameters():
            if param.device != torch.device("cpu"):
                print(f"> {name}: {param.device}")
                num_params += param.numel()
            else:
                print(f"> {name}: {param.device}")
        print(f"RANK {rank} params: {num_params}")
    dist.barrier()
    print()
#print(wrapper_pp.module)