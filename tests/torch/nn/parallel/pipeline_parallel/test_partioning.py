import torch
import torch.distributed as dist
from transformers import T5ForConditionalGeneration

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params

parallel_context = ParallelContext.from_torch(pipeline_parallel_size=4)
model = T5ForConditionalGeneration.from_pretrained("t5-large")

wrapper_pp = PipelineParallel(model, parallel_context=parallel_context)
allocate_params(wrapper_pp, parallel_context)

for rank in range(dist.get_world_size()):
    if dist.get_rank() == rank:
        print(f"RANK: {rank}:")
        num_params = 0
        for name, param in wrapper_pp.named_parameters():
            if param.device != torch.device("cpu"):
                print(f"> {name}: {param.device}")
                num_params += param.numel()
        print(f"RANK {rank} params: {num_params}")
    dist.barrier()
