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

available_devices = ["cpu"]
if torch.cuda.is_available():
    available_devices.append("cuda")

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


def objects_are_equal(a: Any, b: Any, raise_exception: bool = False, dict_key: Optional[str] = None) -> bool:
    """
    Test that two objects are equal. Tensors are compared to ensure matching
    size, dtype, device and values.
    """
    if type(a) is not type(b):
        if raise_exception:
            raise ValueError(f"type mismatch {type(a)} vs. {type(b)}")
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if raise_exception:
                raise ValueError(f"keys mismatch {a.keys()} vs. {b.keys()}")
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception, k):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            if raise_exception:
                raise ValueError(f"length mismatch {len(a)} vs. {len(b)}")
            return False
        return all(objects_are_equal(x, y, raise_exception) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        try:
            # assert_allclose doesn't strictly test shape, dtype and device
            shape_dtype_device_match = a.size() == b.size() and a.dtype == b.dtype and a.device == b.device
            if not shape_dtype_device_match:
                if raise_exception:
                    msg = f"sizes: {a.size()} vs. {b.size()}, "
                    msg += f"types: {a.dtype} vs. {b.dtype}, "
                    msg += f"device: {a.device} vs. {b.device}"
                    raise AssertionError(msg)
                else:
                    return False
            # assert_allclose.
            torch.testing.assert_allclose(a, b)
            return True
        except (AssertionError, RuntimeError) as e:
            if raise_exception:
                if dict_key and isinstance(e, AssertionError):
                    # Add dict key to the assertion error.
                    msg = e.args[0]
                    new_msg = f"For dict key '{dict_key}': {msg}"
                    raise AssertionError(new_msg) from None
                else:
                    raise e
            else:
                return False
    else:
        return a == b

def check_same_model_params(model_a: torch.nn.Module, model_b: torch.nn.Module, message: str = "") -> None:
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-3), f"Model parameters differ\n{p_a} {p_b}\n" + message

    for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
        assert torch.allclose(b_a, b_b), f"Model buffers differ {b_a} - {b_b}\n" + message


def check_same_models_across_ranks(
    model: torch.nn.Module, process_group: Any, params_should_be_equal: bool, check_broadcast_buffers: bool
) -> None:
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    for param in model.parameters():
        # collect the params across the rank
        receptacle = [param.clone() for _ in range(world_size)]
        dist.all_gather(receptacle, param, group=process_group)

        if rank == 0:
            for sync_p in receptacle[1:]:
                assert not params_should_be_equal or torch.all(
                    torch.eq(receptacle[0], sync_p)
                ), f"Models differ in between ranks {receptacle[0]} - {sync_p}"

    # Check that all the buffers are in sync (authoritative rank is 0, its buffer is 0)
    if check_broadcast_buffers:
        for buffer in model.buffers():
            receptacle = [buffer.clone() for _ in range(world_size)]
            dist.all_gather(receptacle, buffer, group=process_group)
            if rank == 0:
                for sync_b in receptacle[1:]:
                    assert not params_should_be_equal or torch.all(
                        torch.eq(receptacle[0], sync_b)
                    ), f"Models differ in between ranks {receptacle[0]} - {sync_b}"

@contextlib.contextmanager
def temp_files_ctx(num: int) -> Generator:
    """A context to get tempfiles and ensure they are cleaned up."""
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    try:
        yield tuple(files)
    finally:
        # temp files could have been removed, so we use rmf.
        for name in files:
            rmf(name)

class SGDWithPausingCompute(torch.optim.SGD):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        self.rank = kwargs["rank"]
        del kwargs["rank"]

        super().__init__(*args, **kwargs)

    def step(self, closure: Optional[Any] = None) -> Any:
        loss = super().step(closure=closure)

        # This is used to make sure that OSS and ShardedDDP enforce a proper stream synchronization
        # - Add a long cuda wait on a compute stream, non blocking from the CPU perspective
        with torch.cuda.stream(torch.cuda.Stream()):
            torch.cuda._sleep(100000000)

            # - optionally change the params on a per rank basis
            with torch.no_grad():
                for param_group in self.param_groups:
                    for param in param_group["params"]:
                        param *= 1.0 + self.rank / 10.0

        return loss

class _Block(Base):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)  # type: ignore
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, *inputs: Any, **kwargs: Any) -> Tensor:
        x = inputs[0]
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x




class GPT2(Base):
    """
    GPT2 pytorch implementation, for testing purposes in the image-GPT context
    Credits: https://github.com/teddykoker/image-gpt"""

    def __init__(
        self, embed_dim: int, num_heads: int, num_layers: int, num_positions: int, num_vocab: int, num_classes: int
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor, classify: bool = False) -> Any:  # type: ignore
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape

        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], dim=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)

        logits = self.head(h)

        if not classify:
            # return logits
            return logits

        h = torch.mean(h, dim=0)  # average pool over sequence
        # return classification logits and generative logits
        return self.clf_head(h), 