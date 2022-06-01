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

torch.set_printoptions(threshold=10_000)

batch_size = 2
sent_len = 4

in_features = 2
out_features = 4
n_layers = 2

world_size = 4
num_experts = world_size
top_k = 1

use_residual = False


# Class for Feed Forward Network
class TestFFNBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Linear(out_features, in_features)

    def forward(self, inp):
        front_out = self.fc1(inp)
        # inter = self.drop_out(self.act(front_out))
        inter = self.act(front_out)
        behind_out = self.fc2(inter)
        behind_out = self.drop_out(behind_out)

        return behind_out


# Class for Entire Model
class TestModel(nn.Module):
    def __init__(self, in_features, out_features, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.in_features = in_features
        self.out_features = out_features

        self.ffns = nn.ModuleList(
            [TestFFNBlock(in_features, out_features) for i in range(n_layers)]
        )

    def forward(self, inp):
        out = inp
        for cur_block in self.ffns:
            out = cur_block(out)
        return out


# Class for Mapping information of Entire Model to expert parallelize
class ExpertParallelMappingForTest(object):
    __MAPPING__ = {"TestModel": [Front("fc1"), Behind("fc2")]}

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
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # 2. Set Parallel Context
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=world_size,
    )

    # 3. Create Model to expert-parallelize
    model_ep = TestModel(in_features, out_features, n_layers)

    # 4. Create Mapping Information used for expert-parallelization
    mapping = ExpertParallelMappingForTest()

    # 5. Wrap Model
    wrapper_ep = ExpertParallel(
        model_ep,
        parallel_context,
        num_experts=num_experts,
        top_k=1,
        use_kernel_optim=False,
        use_residual=use_residual,
        mapping=mapping,
    )

    # 6. Forward Propagation
    token_inp = torch.randn(sent_len, batch_size, in_features).to(f"cuda:{rank}")
    output = wrapper_ep(token_inp)
    print(f"Worker #{rank}'s Output : {output}")

    pred = output.transpose(0, 1)[:, 0].squeeze()
    print(f"pred : {pred}")

    crit = nn.CrossEntropyLoss()
    target = torch.FloatTensor([0, 1, 0, 1]).to(f"cuda:{rank}").view(batch_size, -1)
    loss = crit(pred, target)
    loss.backward()

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
