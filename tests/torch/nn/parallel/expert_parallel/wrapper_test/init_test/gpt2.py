import random
from functools import partial
import os

import numpy as np

import torch
import torch.multiprocessing as mp

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel

from oslo.torch.distributed import ParallelContext, ParallelMode

torch.set_printoptions(threshold=10_000)

num_experts = 4
top_k = 1

use_residual = False

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

    # 3. Create Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Create Model to expert-parallelize
    model_ep = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

    # 5. Wrap Model
    wrapper_ep = ExpertParallel(
        model_ep,
        parallel_context,
        num_experts=num_experts,
        top_k=1,
        use_kernel_optim=False,
        use_residual=use_residual
    )

    # 6. Print the result of wrapping
    print(f'Worker #{rank} : {wrapper_ep.device}')
    print(wrapper_ep)
    print('='*89)

    for param_name, module in wrapper_ep.named_parameters() :
        if wrapper_ep.expert_parallel_mapping.is_front_parallel(wrapper_ep.model, param_name)\
        or wrapper_ep.expert_parallel_mapping.is_behind_parallel(wrapper_ep.model, param_name):
            print(f'Worker #{rank} - param_name : {param_name}, param_size : {module.size()}')
            print(f'Worker #{rank} - param  : {module}')

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


