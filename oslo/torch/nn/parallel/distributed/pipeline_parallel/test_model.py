import torch
from torch import nn
from oslo.torch.nn.parallel.distributed.pipeline_parallel._model_partitioner import (
    ModelPartitioner,
)
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group



class CustomModule(nn.Module):
    def __init__(self, ndim=8):
        super().__init__()
        self.rank = 0
        self.rank_parent = 0
        
        ### module 1
        self.module1 = nn.Linear(ndim, ndim)
        
        setattr(self.module1, "rank", 0)
        setattr(self.module1, "rank_parent", 0) # or should default to None?
        
        ### module 2
        self.module2 = nn.Sequential(
            nn.Linear(ndim, ndim), # rank 0
            nn.Sequential(
                nn.Linear(ndim, ndim), # rank 1
                nn.Linear(ndim, ndim), # rank 0
            ),
            nn.Linear(ndim, ndim), # rank 1
        )
        setattr(self.module2, "rank", 0)
        setattr(self.module2, "rank_parent", 0)
        
        setattr(self.module2[0], "rank", 0)
        setattr(self.module2[0], "rank_parent", 0)
        
        setattr(self.module2[1], "rank", 0)
        setattr(self.module2[1], "rank_parent", 0)
        
        setattr(self.module2[1][0], "rank", 1)
        setattr(self.module2[1][0], "rank_parent", 0) # TO DO: Check this
        
        setattr(self.module2[1][1], "rank", 0)
        setattr(self.module2[1][1], "rank_parent", 0)
        
        setattr(self.module2[2], "rank", 1)
        setattr(self.module2[2], "rank_parent", 0)
        
        
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return x

model = CustomModule()
dist.init_process_group('nccl')

#print(model)

#for name, module in model.named_modules():
#    print(name, module.__class__.__name__, getattr(module, "rank", None), getattr(module, "rank_parent", None))
#    setattr(module, "orig_forward", module.forward)
#    setattr(module, "forward", "SOLUTION")
#print("BEFORE")
#print(model)
#print("\n\n\n\n\n\n\n")
model = ModelPartitioner(model, _get_default_group())
#model = ModelPartitioner(model)
model.partition()
#print("AFTER")
#print(model)
