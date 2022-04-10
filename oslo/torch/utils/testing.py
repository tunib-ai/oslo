# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We're not responsible for pytest decorators
# mypy: disallow_untyped_decorators = False

"""
Collection of some testing utilities for the Fairscale library. Please complement as
you see fit, but refrain from ad-hoc test utils within the different feature sets and
relative imports.
"""

import contextlib
import functools
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.distributed import rpc

from oslo.torch.utils import torch_version

if TYPE_CHECKING:
    Base = nn.Module[Tensor]
else:
    Base = nn.Module

skip_if_cuda = pytest.mark.skipif(torch.cuda.is_available(), reason="Testing only on CPUs to save time")

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1, reason="CUDA required"
)

skip_if_single_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)

skip_if_less_than_four_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason="4 GPUs or more required"
)

skip_if_py38 = pytest.mark.skipif(
    sys.version_info.major == 3 and sys.version_info.minor == 8, reason="Python3.8 is skipped"
)

skip_if_py39_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() and sys.version_info.major == 3 and sys.version_info.minor == 9,
    reason="Python3.9 without CUDA is skipped",
)


def make_cudnn_deterministic() -> None:
    """Make cudnn (matmul) deterministic"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 also make things nondeterministic. Disable it.
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore
    torch.backends.cudnn.allow_tf32 = False  # type: ignore


def dist_init(rank: int, world_size: int, filename: str, filename_rpc: str = "") -> bool:
    """
    Initialize torch distributed, based on a temporary file shared across ranks, which makes it possible for unrelated
    tests to be run concurrently.
    Return false if not enough GPUs present in the system.
    .. warning: This limits the usecase to all ranks being on the same node
    """

    try:
        torch.distributed.rpc.shutdown()
    except Exception:
        pass

    print(f"dist init r={rank}, world={world_size}")

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    url = "file://" + filename
    url_rpc = "file://" + filename_rpc

    if torch_version() >= (1, 6, 0):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            logging.warning("Requested world size cannot be reached on this machine, not enough GPUs")
            return False

        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method=url)

        tp_options = {"init_method": url_rpc}
        if torch_version() == (1, 11, 0):
            if torch.cuda.is_available():
                # Workaround for https://github.com/pytorch/tensorpipe/issues/413
                tp_options["_transports"] = ["shm", "uv"]  # type: ignore

        rpc.init_rpc(
            f"Test{rank}",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(**tp_options),
        )

    else:
        if world_size > 1:
            # TensorPipe is not available in Torch 1.5
            rpc.init_rpc(
                name=f"Test{rank}",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(init_method=url_rpc),
            )
        elif torch.cuda.is_available():
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method=url)
        else:
            return False

    if torch.cuda.is_available() and torch.cuda.device_count():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    return True


def get_worker_map() -> Dict[Any, Any]:
    return {rank: f"Test{rank}" for rank in range(dist.get_world_size())}


def get_world_sizes() -> List[int]:
    limit = torch.cuda.device_count()
    return [x for x in [1, 2, 4, 8] if x <= limit]


def test_runner(
    rank: int, test_func: Callable, deterministic: bool = False, *args: List[Any], **kwargs: Dict[str, Any]
) -> None:
    # At this point we're in a new process, torch options need to be set again
    if deterministic:
        make_cudnn_deterministic()
        torch.manual_seed(1357)

    test_func(rank, *args, **kwargs)


def spawn_for_all_world_sizes(
    test_func: Callable, world_sizes: List[int] = get_world_sizes(), args: Any = [], deterministic: bool = False
) -> None:
    for world_size in world_sizes:
        _, filename = tempfile.mkstemp()
        _, filename_rpc = tempfile.mkstemp()

        try:
            # (lefaudeux) Let mp handle the process joining, join=False and handling context has
            # been unstable in the past.
            mp.spawn(
                test_runner,
                args=(test_func, deterministic, world_size, filename, filename_rpc, *args),
                nprocs=world_size,
                join=True,
            )
        finally:
            rmf(filename)
            rmf(filename_rpc)


def rmf(filename: str) -> None:
    """Remove a file like rm -f."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
