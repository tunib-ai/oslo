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

from init_utils import TestFFNBlock, fix_seed, sequence_dataloader

torch.set_printoptions(threshold=10_000)


batch_size = 2
total_samples = 2
sent_len = 4

hidden_dim = 2
in_features = hidden_dim
out_features = 4

world_size = 2
num_experts = world_size
top_k = 1

ep_size = world_size

use_residual = False


class SimpleMoEModel(torch.nn.Module):
    def __init__(self, linear, moe):
        super().__init__()

        self.linear = linear
        self.moe = moe
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        linear_out = self.linear(x)
        moe_out, _, _ = self.moe(linear_out)

        resid_out = x + moe_out
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
    # ffn_linear1 = torch.nn.Linear(in_features, out_features).to(rank)
    # ffn_linear2 = torch.nn.Linear(out_features, in_features).to(rank)
    # print(f'ffn_linear1.weight : {ffn_linear1.weight}')
    # print(f'ffn_linear2.weight : {ffn_linear2.weight}')

    ffn = TestFFNBlock(in_features, out_features)
    # ffn.fc1 = ffn_linear1
    # ffn.fc2 = ffn_linear2
    moe = MoE(
        in_features,
        expert=ffn,
        ep_size=ep_size,
        use_residual=use_residual,
        num_experts=num_experts,
        k=top_k,
    )
    model = SimpleMoEModel(linear, moe).to(rank)

    for param_name, module in model.named_parameters():
        print(
            f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
        )
        print(f"Worker #{rank} - param  : {module}")


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_expert_parallel_block()
