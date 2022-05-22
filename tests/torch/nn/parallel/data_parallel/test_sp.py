import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import SGD

from oslo.torch.nn.modules.activation import MultiheadAttention
from oslo.torch.nn.parallel.data_parallel.sequence_data_parallel import (
    SequenceDataParallel,
)
from oslo.torch.distributed import ParallelContext, ParallelMode

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def print_rank(s):
    print(f"rank{parallel_context.get_global_rank()}\n {s}")


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


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
        )

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        self.mlp1 = nn.Linear(
            in_features=hidden_size,
            out_features=int(hidden_size * mlp_ratio),
        )
        self.mlp2 = nn.Linear(
            in_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
        )

    def forward(self, hidden_states, attn_mask):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
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
        )
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
        )
        self.layernorm = nn.LayerNorm(hidden_size).double()

    def forward(self, x, attn_mask):
        x = self.emb(x)
        x = self.layer(x, attn_mask)
        x = self.linear(x)
        x = self.layernorm(x)

        # last
        word_embeddings_weight = self.emb.weight
        x = F.linear(x, word_embeddings_weight)
        return x


parallel_context = ParallelContext.from_torch(
    sequence_parallel_size=2,
    data_parallel_size=2,
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


dtype_to_filename = {
    torch.double: "double",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def test_with_dtype(test_dtype, use_mask=False):
    # tested and this makes same result with torch native
    single_model = TesterModel(
        vocab_size=vocab_size,
        hidden_size=d_model,
        num_attention_heads=nhead,
        mlp_ratio=4,
        batch_first=batch_first,
        use_sequence_parallel=False,
        parallel_context=parallel_context,
    ).to(test_dtype)

    # oslo
    multi_model = TesterModel(
        vocab_size=vocab_size,
        hidden_size=d_model,
        num_attention_heads=nhead,
        mlp_ratio=4,
        batch_first=batch_first,
        use_sequence_parallel=True,
        parallel_context=parallel_context,
    ).to(test_dtype)

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

    single_model_losses = []
    multi_model_losses = []
    for i in range(4000):
        dummy_input = (
            torch.randint(vocab_size, (batch_size, sequence_length)).long().cuda()
        )
        dummy_output = (
            torch.randint(vocab_size, (batch_size, sequence_length)).long().cuda()
        )

        if use_mask:
            mask_probs = torch.full(dummy_output.shape, 0.15)
            masked_indices = torch.bernoulli(mask_probs).bool().cuda()
            dummy_output[masked_indices] = -100

            mask = masked_indices.unsqueeze(1) * masked_indices.unsqueeze(2)
        else:
            mask = None

        single_model_input = dummy_input.detach()
        single_model_target = dummy_output.detach()
        if mask is not None:
            single_model_mask = mask.detach()
            # TODO; check reshaping order of MHA
            single_model_mask = single_model_mask.view(
                batch_size, 1, sequence_length, sequence_length
            )
            single_model_mask = single_model_mask.expand(-1, nhead, -1, -1)
            single_model_mask = single_model_mask.reshape(
                batch_size * nhead, sequence_length, sequence_length
            )
        else:
            single_model_mask = None
        single_model_output = single_model(single_model_input, single_model_mask)
        single_model_output_reshaped = single_model_output.reshape(-1, vocab_size)

        # MLM loss is averaged by the number of target tokens
        single_model_loss = loss_fn(
            single_model_output_reshaped, single_model_target.reshape(-1)
        )
        single_num_targets = torch.sum(single_model_target != loss_fn.ignore_index)
        single_model_loss /= single_num_targets

        single_model_optim.zero_grad()
        single_model_loss.backward()
        single_model_optim.step()

        # append loss to plot
        single_model_losses.append(single_model_loss.item())

        # retrieve sub tensor
        sub_batch_size = batch_size // parallel_context.get_world_size(
            ParallelMode.DATA
        )
        batch_start = (
            parallel_context.get_local_rank(ParallelMode.DATA) * sub_batch_size
        )
        batch_end = batch_start + sub_batch_size
        sub_seq_length = sequence_length // parallel_context.get_world_size(
            ParallelMode.SEQUENCE
        )
        start_idx = (
            parallel_context.get_local_rank(ParallelMode.SEQUENCE) * sub_seq_length
        )
        end_idx = start_idx + sub_seq_length
        multi_model_input = dummy_input.detach()[
            batch_start:batch_end, start_idx:end_idx
        ]
        multi_model_target = dummy_output.detach()[
            batch_start:batch_end, start_idx:end_idx
        ]
        if mask is not None:
            # mask need to be the shape of (batch x nhead, target seq len, source seq len)
            multi_model_mask = mask.detach()[batch_start:batch_end, start_idx:end_idx]
            multi_model_mask = multi_model_mask.view(
                sub_batch_size, 1, sub_seq_length, sequence_length
            )
            multi_model_mask = multi_model_mask.expand(-1, nhead, -1, -1)
            multi_model_mask = multi_model_mask.reshape(
                sub_batch_size * nhead, sub_seq_length, sequence_length
            )
        else:
            multi_model_mask = None
        multi_model_output = multi_model(multi_model_input, multi_model_mask)
        multi_model_output_reshaped = multi_model_output.reshape(-1, vocab_size)

        multi_model_loss = loss_fn(
            multi_model_output_reshaped, multi_model_target.reshape(-1)
        )
        # gather the number of target tokens across processes
        multi_num_targets = torch.sum(multi_model_target != loss_fn.ignore_index)
        dist.all_reduce(
            multi_num_targets,
            op=dist.ReduceOp.SUM,
            group=parallel_context.get_group(ParallelMode.SEQUENCE),
        )
        multi_model_loss /= multi_num_targets

        multi_model_optim.zero_grad()
        multi_model_loss.backward()
        multi_model_optim.step()

        # There exists some error since the numbers of
        # target tokens may not be same between DDP processes.
        multi_model_loss = multi_model_loss.detach()
        dist.all_reduce(
            multi_model_loss,
            op=dist.ReduceOp.SUM,
            group=parallel_context.get_group(ParallelMode.SEQUENCE_DP),
        )
        multi_model_loss /= parallel_context.get_world_size(ParallelMode.DATA)
        multi_model_losses.append(multi_model_loss.item())

        # Run accurate test if the testing type is double and without masking
        if test_dtype == torch.double and not use_mask:
            # forward pass test
            assert torch.allclose(
                single_model_output[batch_start:batch_end, start_idx:end_idx],
                multi_model_output,
                atol=5e-7,
            )

            # backward pass test
            for (name, torch_m), (name_, oslo_m) in zip(
                single_model.named_parameters(), multi_model.module.named_parameters()
            ):
                assert name == name_, (name, name_)
                assert torch.allclose(torch_m.grad, oslo_m.grad, atol=5e-7)

    torch.distributed.barrier()

    if parallel_context.get_global_rank() == 0:
        plt.figure(figsize=(64, 16))
        plt.plot(single_model_losses, label="single")
        plt.plot(multi_model_losses, label="multi")
        plt.legend()
        plt.title(f"SP test - {dtype_to_filename[test_dtype]}, mask {use_mask}")
        plt.savefig(f"{dtype_to_filename[test_dtype]}_{use_mask}.png")


if __name__ == "__main__":
    test_with_dtype(torch.double, False)
    test_with_dtype(torch.double, True)
    test_with_dtype(torch.float32, False)
    test_with_dtype(torch.float32, True)
    test_with_dtype(torch.float16, False)
    test_with_dtype(torch.float16, True)
    test_with_dtype(torch.bfloat16, False)
    test_with_dtype(torch.bfloat16, True)
