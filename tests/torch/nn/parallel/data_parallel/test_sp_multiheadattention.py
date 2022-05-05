import torch
from torch.nn.modules.activation import MultiheadAttention as torchMHA

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.modules.activation import MultiheadAttention as osloMHA
from oslo.torch.nn.parallel.data_parallel.sequence_data_parallel import SequenceDataParallel
from oslo.torch.distributed import ParallelContext, ParallelMode

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=2,
    tensor_parallel_mode="sequence",
)


d_model = 512
nhead = 4
dropout = 0.0
batch_first = False

# to make same dummy tensor between processes
torch.manual_seed(0)

# create dummy tensor whose shape is (L, B, E)
sequence_length = 10
dummy_tensor = torch.rand(sequence_length, 32, 512).double().cuda()

# torch native
# torch_mha = torchMHA(
#     d_model,
#     nhead,
#     dropout=dropout,
#     batch_first=batch_first,
#
# ).double()

# tested and this makes same result with torch native
torch_mha = osloMHA(
    d_model,
    nhead,
    dropout=dropout,
    batch_first=batch_first,
    use_sequence_parallel=False,
    parallel_context=parallel_context,
).double()


# oslo
oslo_mha = osloMHA(
    d_model,
    nhead,
    dropout=dropout,
    batch_first=batch_first,
    use_sequence_parallel=True,
    # use_sequence_parallel=False,
    parallel_context=parallel_context,
).double()

# copy weight data
# oslo_mha.in_proj_weight.data.copy_(torch_mha.in_proj_weight.data.copy_)
oslo_mha.load_state_dict(torch_mha.state_dict())

# to cuda
torch_mha.cuda()
oslo_mha.cuda()

# wrap sequence data parallel
oslo_mha_wrapped = SequenceDataParallel(oslo_mha, parallel_context)

torch_mha_output, torch_mha_output_weights = torch_mha(
    dummy_tensor.detach(), dummy_tensor.detach(), dummy_tensor.detach(),
    average_attn_weights=False,
)

# retrieve sub tensor
sub_seq_length = sequence_length // parallel_context.get_world_size(ParallelMode.SEQUENCE)
start_idx = parallel_context.get_local_rank(ParallelMode.SEQUENCE) * sub_seq_length
end_idx = start_idx + sub_seq_length
dummy_tensor = dummy_tensor[start_idx:end_idx]

oslo_mha_output, oslo_mha_output_weights = oslo_mha_wrapped(
    dummy_tensor.detach(), dummy_tensor.detach(), dummy_tensor.detach()
)


def print_rank(s):
    print(f"rank{parallel_context.get_global_rank()} {s}")


# forward pass test
print_rank(f"mha_output: {torch.allclose(torch_mha_output[start_idx:end_idx], oslo_mha_output)}")
print_rank(f"mha_weights: {torch.allclose(torch_mha_output_weights[:, :, start_idx:end_idx], oslo_mha_output_weights)}")
