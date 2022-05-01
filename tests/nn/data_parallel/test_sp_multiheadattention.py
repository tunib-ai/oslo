import torch
from torch.nn.modules.activation import MultiheadAttention as torchMHA

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.modules.activation import MultiheadAttention as osloMHA
from oslo.torch.nn.parallel.distributed.data_parallel.sequence_parallel import (
    SequenceParallel,
)
from oslo.torch.nn.parallel.utils import allocate_params

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=2,
    tensor_parallel_mode="sequence",
)


d_model = 512
nhead = 4
dropout = 0.1
batch_first = False

torch_mha = torchMHA(d_model, nhead, dropout=dropout, batch_first=batch_first).cuda()
oslo_mha = osloMHA(
    d_model,
    nhead,
    dropout=dropout,
    batch_first=batch_first,
    use_sequence_parallel=True,
    parallel_context=parallel_context,
)

allocate_params(oslo_mha, parallel_context)
wrapper = SequenceParallel(oslo_mha, parallel_context)

dummy_tensor = torch.rand(10, 32, 512)
torch_mha_output, torch_mha_output_wegiths = torch_mha(
    dummy_tensor, dummy_tensor, dummy_tensor
)
oslo_mha_output, oslo_mha_output_wegiths = oslo_mha(
    dummy_tensor, dummy_tensor, dummy_tensor
)

print(torch.allclose(torch_mha_output, oslo_mha_output))
print(torch.allclose(torch_mha_output_wegiths, oslo_mha_output_wegiths))
