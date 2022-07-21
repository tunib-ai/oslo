import torch

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.transformers import GPT2LMHeadModel

pc = ParallelContext.from_torch(pipeline_parallel_size=4)
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = PipelineParallel(model, parallel_context=pc)

model(torch.tensor([[1, 2, 3, 4]]).cuda())
