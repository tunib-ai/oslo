import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from oslo.torch.nn.modules.activation import MultiheadAttention
from oslo.torch.nn.parallel.data_parallel.sequence_data_parallel import SequenceDataParallel
from oslo.torch.distributed import ParallelContext, ParallelMode


def print_rank(s):
    print(f"rank{parallel_context.get_global_rank()}\n {s}")


# define simple model
class TesterLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            mlp_ratio,
            use_sequence_parallel,
            batch_first,
            parallel_context,
    ):
        super().__init__()
        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm(hidden_size).double()

        # Self attention.
        self.self_attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=batch_first,
            use_sequence_parallel=use_sequence_parallel,
            parallel_context=parallel_context,
        ).double()

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(hidden_size).double()

        self.mlp1 = nn.Linear(
            in_features=hidden_size,
            out_features=int(hidden_size * mlp_ratio),
        ).double()
        self.mlp2 = nn.Linear(
            in_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
        ).double()

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, _ = self.self_attention(x, x, x)
        x = x + residual
        x = F.gelu(x)
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x) + residual
        x = F.gelu(x)
        return x


class TesterModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            num_attention_heads,
            mlp_ratio,
            batch_first,
            use_sequence_parallel,
            parallel_context,
    ):
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        ).double()
        self.layer = TesterLayer(
            hidden_size,
            num_attention_heads,
            mlp_ratio,
            use_sequence_parallel,
            batch_first,
            parallel_context,
        )

        # head
        self.linear = nn.Linear(
            hidden_size,
            hidden_size,
        ).double()
        self.layernorm = nn.LayerNorm(hidden_size).double()

    def forward(self, x):
        x = self.emb(x)
        x = self.layer(x)
        x = self.linear(x)
        x = self.layernorm(x)

        # last
        word_embeddings_weight = self.emb.weight
        x = F.linear(x, word_embeddings_weight)
        return x


parallel_context = ParallelContext.from_torch(
    data_parallel_size=2,
    pipeline_parallel_size=1,
    tensor_parallel_size=2,
    tensor_parallel_mode="sequence",
)

d_model = 512
nhead = 4
dropout = 0.0
vocab_size = 5000
batch_first = True

# to make same dummy tensor between processes
torch.manual_seed(0)

# create dummy tensor whose shape is (L, B, E)
batch_size = 32
sequence_length = 10

# tested and this makes same result with torch native
single_model = TesterModel(
    vocab_size=vocab_size,
    hidden_size=d_model,
    num_attention_heads=nhead,
    mlp_ratio=4,
    batch_first=batch_first,
    use_sequence_parallel=False,
    parallel_context=parallel_context,
).double()

# oslo
multi_model = TesterModel(
    vocab_size=vocab_size,
    hidden_size=d_model,
    num_attention_heads=nhead,
    mlp_ratio=4,
    batch_first=batch_first,
    use_sequence_parallel=True,
    parallel_context=parallel_context,
).double()

# copy weight data
multi_model.load_state_dict(single_model.state_dict())

# to cuda
single_model.cuda()
multi_model.cuda()

# define loss
loss_fn = nn.CrossEntropyLoss(reduction="sum")

# define optimizer
lr = 1e-4
single_model_optim = SGD(single_model.parameters(), lr=lr)
multi_model_optim = SGD(multi_model.parameters(), lr=lr)

# wrap sequence data parallel
multi_model = SequenceDataParallel(multi_model, parallel_context)


for i in range(1000):
    dummy_input = torch.randint(vocab_size, (batch_size, sequence_length)).long().cuda()
    dummy_output = torch.randint(vocab_size, (batch_size, sequence_length)).long().cuda()

    single_model_input = dummy_input.detach()
    single_model_target = dummy_output.detach()
    single_model_output = single_model(single_model_input)
    single_model_output_reshaped = single_model_output.reshape(-1, vocab_size)
    single_model_loss = loss_fn(single_model_output_reshaped, single_model_target.reshape(-1))
    single_model_loss /= batch_size
    single_model_optim.zero_grad()
    single_model_loss.backward()
    single_model_optim.step()

    # retrieve sub tensor
    sub_batch_size = batch_size // parallel_context.get_world_size(ParallelMode.DATA)
    batch_start = parallel_context.get_local_rank(ParallelMode.DATA) * sub_batch_size
    batch_end = batch_start + sub_batch_size
    sub_seq_length = sequence_length // parallel_context.get_world_size(ParallelMode.SEQUENCE)
    start_idx = parallel_context.get_local_rank(ParallelMode.SEQUENCE) * sub_seq_length
    end_idx = start_idx + sub_seq_length
    multi_model_input = dummy_input.detach()[batch_start:batch_end, start_idx:end_idx]
    multi_model_target = dummy_output.detach()[batch_start:batch_end, start_idx:end_idx]
    multi_model_output = multi_model(multi_model_input)
    multi_model_output_reshaped = multi_model_output.reshape(-1, vocab_size)
    multi_model_loss = loss_fn(multi_model_output_reshaped, multi_model_target.reshape(-1))
    multi_model_loss /= sub_batch_size      # average by batch size since comm. hook deals with this
    multi_model_optim.zero_grad()
    multi_model_loss.backward()
    multi_model_optim.step()

    # forward pass test
    assert torch.allclose(single_model_output[batch_start:batch_end, start_idx:end_idx], multi_model_output, atol=5e-7)

    # backward pass test
    for (name, torch_m), (name_, oslo_m) in zip(single_model.named_parameters(),
                                                multi_model.module.named_parameters()):
        assert name == name_, (name, name_)
        assert torch.allclose(torch_m.grad, oslo_m.grad, atol=5e-7)


# clean up ?
torch.distributed.barrier()
