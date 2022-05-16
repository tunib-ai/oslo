import random
from functools import partial
import os

import numpy as np

import torch
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext

from oslo.torch.distributed._seed.helper import _SEED_MANAGER, seed

from oslo.torch.distributed import ParallelContext, ParallelMode

torch.set_printoptions(threshold=10_000)

batch_size = 2
sent_len = 4
token_num = 7
in_features = 2
out_features = 4
num_experts = 2
top_k = 1


def run_test(rank, port):
    # 1. Generate Input
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 2. Create ExpertParallelFrontBlock
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=2,
    )
    ep_context = ExpertParallelContext(parallel_context, 2)
    ep_context.setup(1000)
    num_local_experts, ep_info = ep_context.get_info(num_experts)

    global_rank = parallel_context.get_global_rank()
    print(f"ep_local_rank : {ep_info.ep_local_rank}")
    print(f"dp_local_rank : {ep_info.dp_local_rank}")
    print(f"GLOBAL RANK : {global_rank}, Num Local Experts : {num_local_experts}")
    print(f"_SEED_MANAGER._seeds : {_SEED_MANAGER._seeds}")
    print(f"_SEED_MANAGER._seed_states : {_SEED_MANAGER._seed_states}")
    import torch.nn as nn

    weight = nn.Parameter(
        torch.empty(1, in_features, out_features, device=f"cuda:{rank}").contiguous(),
    )
    import math

    std = math.sqrt(0.1 / 2)
    with seed(ParallelMode.TENSOR):
        nn.init.trunc_normal_(weight, std=std)

    print(f"weight : {weight}")

    return


def test_expert_parallel_block():
    world_size = 2
    run_func = partial(run_test, port=29500)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # 1. Set Random Seed for Reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_expert_parallel_block()
