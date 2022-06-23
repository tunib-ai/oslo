import random
from functools import partial
import os

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.expert_parallel.layers import (
    ExpertParallelFrontBlock,
    ExpertParallelFrontBlockDS,
    ExpertParallelBehindBlock,
    ExpertParallelBehindBlockDS,
    TopKGate,
)
from oslo.torch.nn.parallel.expert_parallel.experts import Experts
from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.distributed._seed.helper import seed

torch.set_printoptions(threshold=10_000)

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


def run_test(rank, port):
    # 1. Configure for Parallelization
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 2. Set Parallel Context
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=2,
    )

    # 3. Set Expert Parallel Context
    ep_context = ExpertParallelContext(parallel_context, use_kernel_optim=False)
    ep_context.setup(3)
    ep_context.reset_loss()
    num_local_experts, ep_info = ep_context.get_info(num_experts)

    global_rank = parallel_context.get_global_rank()
    print(f"ep_local_rank : {ep_info.ep_local_rank}")
    print(f"dp_local_rank : {ep_info.dp_local_rank}")
    print(f"GLOBAL RANK : {global_rank}, Num Local Experts : {num_local_experts}")

    # 4. Create ExpertParallelFrontBlock
    link_info = dict()
    gate = TopKGate(
        in_features,
        num_experts,
        top_k,
        capacity_factor_train,
        capacity_factor_eval,
        min_capacity,
        noisy_policy,
        drop_tokens,
        use_rts=use_rts,
    )
    weight = torch.Tensor([[0.5406, 0.5869], [-0.1657, 0.6496]])
    gate.wg.weight = torch.nn.Parameter(data=weight)

    expert_parallel_residual = nn.Linear(in_features, out_features, bias=None)
    weight = torch.Tensor(
        [[-0.0515, 0.4879], [0.4020, -0.0062], [0.0636, 0.1600], [0.1424, 0.6623]]
    )
    expert_parallel_residual.weight = torch.nn.Parameter(data=weight)

    expert_parallel_residual_mix = nn.Linear(in_features, 2, bias=None)
    weight = torch.Tensor([[0.3972, -0.0215], [-0.1624, -0.5637]])
    expert_parallel_residual_mix.weight = torch.nn.Parameter(data=weight)
    bias = torch.Tensor([0.4714, 0.0398])
    expert_parallel_residual_mix.bias = torch.nn.Parameter(data=bias)

    front_expert = nn.Linear(in_features, out_features, bias=None)
    weight = torch.tensor(
        [[-0.0515, 0.4879], [0.4020, -0.0062], [0.0636, 0.1600], [0.1424, 0.6623]]
    )
    front_expert.weight = torch.nn.Parameter(data=weight)

    front_experts = Experts(front_expert, num_local_experts)

    num_local_experts, ep_info = ep_context.get_info(num_experts)
    front = ExpertParallelFrontBlockDS(
        ep_context,
        link_info,
        in_features,
        out_features,
        gate,
        front_experts,
        ep_info.ep_size,
        num_local_experts,
        use_residual,
        expert_parallel_residual,
    )
    front.expert_parallel_residual = expert_parallel_residual
    front.expert_parallel_residual_mix = expert_parallel_residual_mix
    front._set_ep_group(ep_info.ep_group)
    front.to(rank)

    activation = nn.ReLU()
    drop_out = nn.Dropout(p=0.1)

    # 5. Create ExpertParallelBehindBlock
    expert_parallel_residual = nn.Linear(out_features, in_features, bias=None)
    weight = torch.Tensor(
        [[-0.1153, 0.3507, -0.1316, -0.4316], [0.4138, -0.3855, -0.3800, -0.3188]]
    )
    expert_parallel_residual.weight = torch.nn.Parameter(data=weight)

    behind_expert = nn.Linear(out_features, in_features, bias=None)
    weight = torch.tensor(
        [[-0.1153, 0.3507, -0.1316, -0.4316], [0.4138, -0.3855, -0.3800, -0.3188]]
    )
    behind_expert.weight = torch.nn.Parameter(data=weight)

    behind_experts = Experts(behind_expert, num_local_experts)

    num_local_experts, ep_info = ep_context.get_info(num_experts)
    behind = ExpertParallelBehindBlockDS(
        ep_context,
        link_info,
        out_features,
        in_features,
        behind_experts,
        ep_info.ep_size,
        num_local_experts,
        use_residual,
        expert_parallel_residual,
    )
    behind._set_ep_group(ep_info.ep_group)
    behind.to(rank)

    # 6. Forward Propagation
    token_inp = (
        torch.arange(sent_len * batch_size * in_features)
        .view(sent_len, batch_size, in_features)
        .to(torch.float32)
        .to(rank)
    )
    print(f"INPUT : {token_inp}")
    front_res = front(token_inp)
    print(f"link_info : {link_info}")

    # with seed(ParallelMode.TENSOR):
    print("=" * 89)
    print("MIDDLE STEP")
    print(f"ACTIVATION INPUT : {front_res}")
    act_out = activation(front_res)
    print(f"ACTIVATION OUTPUT : {act_out}")
    dropped = drop_out(act_out)
    print(f"DROP OUT OUTPUT : {dropped}")
    print("=" * 89)
    print("BEHIND STEP")
    # print(f"BEHIND INPUT  : {dropped}")
    print(f"BEHIND INPUT  : {act_out}")
    # behind_res = behind(dropped)
    behind_res = behind(act_out)
    print(f"BEHIND RESULT : {behind_res}")
    print("=" * 89)

    return


def test_expert_parallel_block():
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # Set Random Seed for Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
