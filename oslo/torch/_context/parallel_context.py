import os
import random

import numpy as np
import torch
import torch.distributed as dist

from oslo.torch._context.initializers.initializer_data import (
    DataParallelGroupInitializer,
)
from oslo.torch._context.initializers.initializer_model import (
    ModelParallelGroupInitializer,
)
from oslo.torch._context.initializers.initializer_pipeline import (
    PipelineParallelGroupInitializer,
)
from oslo.torch._context.initializers.initializer_sequence import (
    SequenceParallelGroupInitializer,
)
from oslo.torch._context.initializers.initializer_tensor import (
    TensorParallelGroupInitializer,
)
from oslo.torch._context.initializers.initializer_tensor_1d import (
    TensorParallel1DGroupInitializer,
)
from oslo.torch._context.initializers.initializer_tensor_2d import (
    TensorParallel2DGroupInitializer,
)
from oslo.torch._context.initializers.initializer_tensor_2p5d import (
    TensorParallel2p5DGroupInitializer,
)
from oslo.torch._context.initializers.initializer_tensor_3d import (
    TensorParallel3DGroupInitializer,
)
from oslo.torch._context.parallel_mode import ParallelMode
from oslo.torch._context.random._helper import add_seed, set_mode
from oslo.torch._context.singleton_meta import SingletonMeta

TensorParallelGroupInitializerByMode = {
    None: None,
    "1d": TensorParallel1DGroupInitializer,
    "2d": TensorParallel2DGroupInitializer,
    "2.5d": TensorParallel2p5DGroupInitializer,
    "3d": TensorParallel3DGroupInitializer,
    "sequence": SequenceParallelGroupInitializer,
}


class ParallelContext(metaclass=SingletonMeta):
    @classmethod
    def from_torch(
        cls,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: str = None,
        tensor_parallel_depth: int = None,
        backend: str = "nccl",
        seed: bool = 42,
    ):
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_depth=tensor_parallel_depth,
            backend=backend,
            seed=seed,
        )

    @classmethod
    def from_slurm(
        cls,
        host: str,
        port: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: str = None,
        tensor_parallel_depth: int = None,
        backend: str = "nccl",
        seed: bool = 42,
        local_rank: int = None,
    ):
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_depth=tensor_parallel_depth,
            backend=backend,
            seed=seed,
        )

    @classmethod
    def from_openmpi(
        cls,
        host: str,
        port: int,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: str = None,
        tensor_parallel_depth: int = None,
        backend: str = "nccl",
        seed: bool = 42,
    ):
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            host=host,
            port=port,
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_depth=tensor_parallel_depth,
            backend=backend,
            seed=seed,
        )

    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        host: str,
        port: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        tensor_parallel_mode: str,
        tensor_parallel_depth: int,
        backend: str,
        seed: int,
    ):
        assert self.tensor_parallel_mode in TensorParallelGroupInitializerByMode, (
            f"param `tensor_parallel_mode` {tensor_parallel_mode} is not available. "
            f"currently, we supports {TensorParallelGroupInitializerByMode.keys()}."
        )

        if self.tensor_parallel_size > 1:
            assert self.tensor_parallel_mode is not None, (
                "param `tensor_parallel_mode` must not be None "
                "if param `tensor_parallel_size` > 1."
            )

        if self.tensor_parallel_mode == "2.5d":
            assert self.tensor_parallel_depth is not None, (
                "param `tensor_parallel_depth` must not be None "
                "if param `tensor_parallel_mode` is '2.5d'."
            )

        assert (
            world_size
            == data_parallel_size * pipeline_parallel_size * tensor_parallel_size
        ), (
            f"Expected the world size {world_size} to be equal to data"
            f" parallel size ({data_parallel_size}) * pipeline parallel size "
            f"({pipeline_parallel_size}) * tensor parallel size ({tensor_parallel_size})."
        )

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._cpu_groups = {}
        self._ranks_in_group = {}

        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_mode = (
            tensor_parallel_mode.lower()
            if isinstance(tensor_parallel_mode, str)
            else None
        )
        # tesseract depth for 2.5d parallelism
        self.tensor_parallel_depth = tensor_parallel_depth

        self.init_global_dist(rank, world_size, backend, host, port)
        self.init_parallel_groups()

        if torch.cuda.is_available():
            # if local rank is not given, calculate automatically
            self.set_device(local_rank)

        self.set_seed(seed)

    # sanity check
    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        assert isinstance(parallel_mode, ParallelMode)

    # world sizes
    def get_world_size(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        self._check_parallel_mode(parallel_mode)
        self._world_sizes[parallel_mode] = world_size

    # local ranks
    def get_local_rank(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks[parallel_mode]

    def add_local_rank(self, parallel_mode: ParallelMode, rank: int):
        self._check_parallel_mode(parallel_mode)
        self._local_ranks[parallel_mode] = rank

    # global ranks
    def get_global_rank(self):
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        self._check_parallel_mode(parallel_mode)
        self._global_ranks[parallel_mode] = rank

    def get_next_global_rank(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)

        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)

        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_first_rank(self, parallel_mode: ParallelMode):
        return self.get_local_rank(parallel_mode) == 0

    def is_last_rank(self, parallel_mode):
        return (
            self.get_local_rank(parallel_mode) == self.get_world_size(parallel_mode) - 1
        )

    # groups
    def get_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        self._check_parallel_mode(parallel_mode)
        self._groups[parallel_mode] = group

    # cpu groups
    def get_cpu_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def add_cpu_group(self, parallel_mode, group: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        self._cpu_groups[parallel_mode] = group

    # ranks in group
    def get_ranks_in_group(self, parallel_mode: ParallelMode):
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def add_ranks_in_group(self, parallel_mode: ParallelMode, ranks: list):
        self._check_parallel_mode(parallel_mode)
        self._ranks_in_group[parallel_mode] = ranks

    # init distributed groups
    def init_global_dist(
        self,
        rank: int,
        world_size: int,
        backend: str,
        host: str,
        port: int,
    ):
        init_method = f"tcp://{host}:{port}"
        dist.init_process_group(
            rank=rank, world_size=world_size, backend=backend, init_method=init_method
        )

        ranks = list(range(world_size))
        cpu_group = (
            dist.new_group(ranks, backend="gloo")
            if dist.get_backend() != "gloo"
            else None
        )
        self._register_dist(
            rank, world_size, None, cpu_group, ranks, ParallelMode.GLOBAL
        )
        self.add_global_rank(ParallelMode.GLOBAL, rank)

    def _register_dist(
        self,
        local_rank: int,
        world_size: int,
        process_group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        ranks_in_group: list,
        mode: ParallelMode,
    ):
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, world_size)
        self.add_group(mode, process_group)
        self.add_cpu_group(mode, cpu_group)
        self.add_ranks_in_group(mode, ranks_in_group)

    def init_parallel_groups(self):
        rank = self.get_global_rank()
        world_size = self.get_world_size(ParallelMode.GLOBAL)
        initializer_param = {
            "rank": rank,
            "world_size": world_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

        initializer_results = [
            DataParallelGroupInitializer(**initializer_param).init_dist_group(),
            ModelParallelGroupInitializer(**initializer_param).init_dist_group(),
            TensorParallelGroupInitializer(**initializer_param).init_dist_group(),
        ]

        tensor_parallel_initializer_cls = TensorParallelGroupInitializerByMode[
            self.tensor_parallel_mode
        ]

        if tensor_parallel_initializer_cls is not None:
            _initializer_param = initializer_param.copy()

            if self.tensor_parallel_mode == "2.5d":
                _initializer_param["depth"] = self.tensor_parallel_depth

            tensor_parallel_initializer = tensor_parallel_initializer_cls(
                **_initializer_param
            )
            initializer_results.append(tensor_parallel_initializer.init_dist_group())

        if self.pipeline_parallel_size > 1:
            initializer_results.append(
                PipelineParallelGroupInitializer(**initializer_param).init_dist_group()
            )

        for initializer_result in initializer_results:
            if isinstance(initializer_result, list):
                for res in initializer_result:
                    self._register_dist(**res)
            else:
                self._register_dist(**initializer_result)

    def is_initialized(self, parallel_mode: ParallelMode):
        return parallel_mode in self._groups

    def destroy(self):
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)

        dist.destroy_process_group()
        self._groups.clear()

    def set_device(self, device_ordinal: int):
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node
        torch.cuda.set_device(device_ordinal)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            # create random seed for different parallel modes
            # data parallel seed are kept the same
            parallel_seed = seed
            add_seed(ParallelMode.DATA, parallel_seed)

            # model parallel seeds are different across ranks
            pipeline_offset = self._local_ranks.get(ParallelMode.PIPELINE, 0)

            # add seed for data parallel and tensor parallel only
            if self.is_initialized(ParallelMode.TENSOR):
                tp_rank = self.get_local_rank(ParallelMode.TENSOR)

                # 100 is only to increase the diff in seeds between pipeline stages
                tp_rank_with_offset = tp_rank + pipeline_offset * 1024
                tp_seed = seed + tp_rank_with_offset
                add_seed(ParallelMode.TENSOR, tp_seed)

            set_mode(ParallelMode.DATA)
