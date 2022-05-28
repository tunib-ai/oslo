import random
from functools import partial
import os

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.expert_parallel.layers import (
    ExpertParallelFrontBlock,
    ExpertParallelBehindBlock,
)
from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.distributed._seed.helper import seed

torch.set_printoptions(threshold=10_000)

batch_size = 2
sent_len = 4
in_features = 2
out_features = 4
num_experts = 2
top_k = 1

use_residual = True


def run_test(rank, port):
    # 1. Configure for Parallelization
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
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
    front = ExpertParallelFrontBlock(
        ep_context=ep_context,
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        link_info=link_info,
        top_k=top_k,
        noisy_policy="Jitter",
        use_residual=use_residual,
    )
    front.to(rank)
    # print(f'[{global_rank}] front weight : {front.weight}')

    activation = nn.GELU()
    drop_out = nn.Dropout(p=0.1)

    # 5. Create ExpertParallelBehindBlock
    behind = ExpertParallelBehindBlock(
        ep_context=ep_context,
        in_features=out_features,
        out_features=in_features,
        num_experts=num_experts,
        link_info=link_info,
        use_residual=use_residual,
    )
    behind.to(rank)

    # 6. Forward Propagation
    token_inp = (
        torch.empty(sent_len, batch_size, in_features).to(torch.float32).to(rank)
    )
    # with seed(ParallelMode.TENSOR):
    #    torch.nn.init.trunc_normal_(token_inp, std=1)
    print(f"INPUT : {token_inp}")
    front_res = front(token_inp)
    print(f"link_info : {link_info}")

    with seed(ParallelMode.TENSOR):
        print("=" * 89)
        print("MIDDLE STEP")
        print(f"ACTIVATION INPUT : {front_res}")
        act_out = activation(front_res)
        resid_act_out = activation(link_info["residual_inter"])
        print(f"ACTIVATION OUTPUT : {act_out}")
        print(f"RESIDUAL ACT OUT : {resid_act_out}")
        dropped = drop_out(act_out)
        resid_dropped = drop_out(resid_act_out)
        print(f"DROP OUT OUTPUT : {dropped}")
        print(f"RESIDUAL DROP OUT : {resid_dropped}")
        print("=" * 89)
    print("BEHIND STEP")
    print(f"BEHIND INPUT  : {dropped}")
    behind_res = behind(dropped)
    print(f"BEHIND RESULT : {behind_res}")
    print("=" * 89)

    return


def test_expert_parallel_block():
    world_size = 2
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
