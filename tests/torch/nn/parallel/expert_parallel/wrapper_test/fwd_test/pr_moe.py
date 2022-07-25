import os
import json
import math
import random
import argparse
from functools import partial

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import deepspeed
from deepspeed.utils import groups
from deepspeed.moe.layer import MoE

from fwd_utils import TestFFNBlock, fix_seed, sequence_dataloader

torch.set_printoptions(threshold=10_000)


batch_size = 2
total_samples = 50
sent_len = 4

hidden_dim = 2
in_features = hidden_dim
out_features = 4

world_size = 2
num_experts = world_size
top_k = 1

ep_size = world_size

use_residual = True


class SimplePRMoEModel(torch.nn.Module):
    def __init__(self, linear, moe1, moe2):
        super().__init__()

        self.linear = linear
        self.moe1 = moe1
        self.moe2 = moe2
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        linear_out = self.linear(x)
        moe_out, _, _ = self.moe1(linear_out)
        moe_out, _, _ = self.moe2(moe_out)

        resid_out = linear_out + moe_out
        sent_emb = resid_out.mean(1)

        return self.cross_entropy_loss(sent_emb, y)


def run_test(rank, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    fix_seed(rank)

    linear = torch.nn.Linear(in_features, in_features)
    ffn1 = TestFFNBlock(in_features, out_features)
    ffn2 = TestFFNBlock(in_features, out_features)
    coef1 = torch.nn.Linear(in_features, in_features)
    coef2 = torch.nn.Linear(in_features, in_features)
    moe1 = MoE(
        in_features,
        expert=ffn1,
        ep_size=ep_size,
        use_residual=use_residual,
        num_experts=num_experts,
        k=top_k,
        use_rts=False,
    )
    moe2 = MoE(
        in_features,
        expert=ffn2,
        ep_size=ep_size,
        use_residual=use_residual,
        num_experts=num_experts * 2,
        k=top_k,
        use_rts=False,
    )
    moe1.coefficient = coef1
    moe2.coefficient = coef2
    model = SimplePRMoEModel(linear, moe1, moe2).to(rank)

    optimizer = torch.optim.AdamW(params=model.parameters())

    data_loader = sequence_dataloader(
        batch_size,
        total_samples,
        hidden_dim=hidden_dim,
        device=rank,
        seq_len=sent_len,
        dtype=torch.float32,
    )
    # for param_name, module in model.named_parameters():
    #    print(
    #        f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
    #    )
    #    print(f"Worker #{rank} - param  : {module}")

    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        print(f"Worker # {rank} Instance #{n} loss : {loss}")
        loss.backward()
        optimizer.step()


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_expert_parallel_block()
