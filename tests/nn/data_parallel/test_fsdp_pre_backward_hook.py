# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with pre-backward hook bug. """

import os
import random

import pytest
import torch
from torch.nn import Linear, Module

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel.distributed import FullyShardedDataParallel as FSDP
from oslo.torch.utils.testing import (
    dist_init,
    skip_if_no_cuda,
    teardown,
    temp_files_ctx,
)


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    # dist_init needs 2 files
    with temp_files_ctx(2) as files:
        yield files


@skip_if_no_cuda
def test_pre_backward_hook(temp_files):
    """Test FSDP with a model that triggers a pre_backward hook bug."""

    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(2000, 3000))

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1
    )
    assert parallel_context, "Dist init failed"

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(4, 4).cuda()
            self.l2 = FSDP(Linear(4, 4).cuda(), parallel_context)
            self.l3 = Linear(4, 4).cuda()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            inner_result = x
            x = self.l3(x)
            return x, inner_result

        def assert_and_clear_grad(self):
            for p in self.parameters():
                assert p.shape in [(4, 4), (4,), (4 * 4 + 4,)], p.shape
                assert p.grad is not None
                p.grad = None

    model = FSDP(Model(), parallel_context, flatten_parameters=False).cuda()
    in_data = torch.rand(1, 4).cuda()
    for _ in range(3):
        out, _ = model(in_data)
        out.sum().backward()
        model.assert_and_clear_grad()

    teardown()
