from oslo.torch.distributed import ParallelContext
from oslo.torch.distributed.nn.functional import send, recv
import torch.distributed as dist

parallel_context = ParallelContext.from_torch(pipeline_parallel_size=2)

example_data = [
    True,
    None,
    1,
    2.3,
    "안녕",
    {"xx": "yy"},
    {"1", "2", "3"},
    (1, 2, 3),
    complex(1, 2),
    [1, 2, [1, 2, {"1": "x", "2": (1, 2, {3})}]],
]

send(example_data, src_rank=0, dst_rank=1, parallel_context=parallel_context)
data = recv(src_rank=0, dst_rank=1, parallel_context=parallel_context)

if dist.get_rank() == 1:
    print(data)
