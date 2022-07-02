import random
from functools import partial
import os

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from oslo.torch.nn.parallel.expert_parallel.expert_parallel import ExpertParallel

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.expert_parallel.mapping import Front, Behind

from init_utils import TestFFNBlock, fix_seed, sequence_dataloader

torch.set_printoptions(threshold=10_000)

total_samples = 2

batch_size = 2
sent_len = 4

hidden_dim = 2
in_features = hidden_dim
out_features = 4
n_layers = 1

world_size = 2
num_experts = world_size
top_k = 1

use_residual = False


class TestMoE(torch.nn.Module):
    def __init__(self, ffns):
        super().__init__()

        self.ffns = nn.ModuleList(ffns)

    def forward(self, x):
        out = x
        for cur_layer in self.ffns:
            out = cur_layer(out)
        return out


class SimpleMoEModel(torch.nn.Module):
    def __init__(self, linear, moe):
        super().__init__()

        self.linear = linear
        self.moe = moe
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        linear_out = self.linear(x)
        moe_out = self.moe(linear_out)

        resid_out = x + moe_out
        sent_emb = resid_out.mean(1)

        return self.cross_entropy_loss(sent_emb, y)


# Class for Mapping information of Entire Model to expert parallelize
class ExpertParallelMappingForTest(object):
    __MAPPING__ = {
        "TestMoE": [
            Front("fc1", enc_name="ffns", layer="ffns"),
            Behind("fc2", enc_name="ffns", layer="ffns"),
        ]
    }

    def __init__(self):
        cache_mapping = {}
        import sys

        for cls_name, mapping in self.__MAPPING__.items():
            cls = globals()[cls_name]
            if cls is not None:
                cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping

    def get_mapping(self, model):
        mapping_by_model = None
        for cls, mapping in self.__MAPPING__.items():
            if isinstance(model, cls):
                mapping_by_model = {cls: mapping}

        assert mapping_by_model is not None, (
            f"Currently, {model.__class__.__qualname__} is not supported. "
            f"The current supported models are {list(self.__MAPPING__.keys())}"
        )
        return mapping_by_model


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
        expert_parallel_size=world_size,
    )
    fix_seed(rank)

    linear = torch.nn.Linear(in_features, in_features)
    # ffn_linear1 = torch.nn.Linear(in_features, out_features).to(rank)
    # ffn_linear2 = torch.nn.Linear(out_features, in_features).to(rank)
    # print(f'ffn_linear1.weight : {ffn_linear1.weight}')
    # print(f'ffn_linear2.weight : {ffn_linear2.weight}')
    ffns = [TestFFNBlock(in_features, out_features) for i in range(n_layers)]
    # for cur in ffns:
    #    cur.fc1 = ffn_linear1
    #    cur.fc2 = ffn_linear2
    moe = TestMoE(ffns)
    mapping = ExpertParallelMappingForTest()
    moe = ExpertParallel(
        moe,
        parallel_context,
        num_enc_experts=num_experts,
        top_k=top_k,
        use_kernel_optim=False,
        use_residual=use_residual,
        mapping=mapping,
    )

    model_ep = SimpleMoEModel(linear, moe).to(rank)

    for param_name, module in model_ep.named_parameters():
        print(
            f"Worker #{rank} - param_name : {param_name}, param_size : {module.size()}"
        )
        print(f"Worker #{rank} - param  : {module}")

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
