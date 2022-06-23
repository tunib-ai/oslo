import random
import numpy as np
from functools import partial
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import torch.distributed as dist

import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import TopKGate

batch_size = 3
sent_len = 4

in_features = 2
out_features = 4
num_experts = 2
top_k = 1

capacity_factor_train = 1.0
capacity_factor_eval = 1.0
min_capacity = 4
noisy_policy = None

drop_tokens = True
use_residual = True

use_rts = False

world_size = 2


class TestFFN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features, bias=False)

    def forward(self, inp):
        print(f"FFN INP : {inp}")
        inter = self.fc1(inp)
        print(f"FFN FC1 WEIGHT : {self.fc1.weight}")
        print(f"FFN INTER : {inter}")
        act_out = self.act(inter)
        print(f"FFN ACT OUT : {act_out}")
        print(f"FFN FC2 WEIGHT : {self.fc2.weight}")
        output = self.fc2(act_out)
        print(f"FFN FC2 OUTPUT : {output}")

        return output


def run_test(rank, port):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    expert = TestFFN(in_features, out_features)

    weight = torch.tensor(
        [[-0.0515, 0.4879], [0.4020, -0.0062], [0.0636, 0.1600], [0.1424, 0.6623]]
    )
    expert.fc1.weight = torch.nn.Parameter(data=weight)

    weight = torch.tensor(
        [[-0.1153, 0.3507, -0.1316, -0.4316], [0.4138, -0.3855, -0.3800, -0.3188]]
    )
    expert.fc2.weight = torch.nn.Parameter(data=weight)

    moe = MoE(
        hidden_size=in_features,
        expert=expert,
        num_experts=num_experts,
        ep_size=2,
        k=top_k,
        capacity_factor=capacity_factor_train,
        eval_capacity_factor=capacity_factor_train,
        min_capacity=min_capacity,
        use_residual=use_residual,
        noisy_gate_policy=noisy_policy,
        drop_tokens=drop_tokens,
        use_rts=use_rts,
    ).to(rank)

    weight = torch.Tensor([[0.5406, 0.5869], [-0.1657, 0.6496]]).to(rank)
    moe.deepspeed_moe.gate.wg.weight = torch.nn.Parameter(data=weight)

    weight = torch.Tensor(
        [[-0.0515, 0.4879], [0.4020, -0.0062], [0.0636, 0.1600], [0.1424, 0.6623]]
    ).to(rank)
    moe.mlp.fc1.weight = torch.nn.Parameter(data=weight)

    weight = torch.Tensor(
        [[-0.1153, 0.3507, -0.1316, -0.4316], [0.4138, -0.3855, -0.3800, -0.3188]]
    ).to(rank)
    moe.mlp.fc2.weight = torch.nn.Parameter(data=weight)

    weight = torch.Tensor([[0.3972, -0.0215], [-0.1624, -0.5637]]).to(rank)
    moe.coefficient.weight = torch.nn.Parameter(data=weight)
    bias = torch.Tensor([0.4714, 0.0398]).to(rank)
    moe.coefficient.bias = torch.nn.Parameter(data=bias)
    print(f"moe.coefficient.bias : {moe.coefficient.bias}")

    token_inp = (
        torch.arange(sent_len * batch_size * in_features)
        .view(sent_len, batch_size, in_features)
        .to(torch.float32)
        .to(rank)
    )
    print(f"INPUT : {token_inp}")
    moe(token_inp)


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
