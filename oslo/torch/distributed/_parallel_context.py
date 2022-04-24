import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from oslo.torch.distributed._initializers.initializer_data import (
    DataParallelGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_model import (
    ModelParallelGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_pipeline import (
    PipelineParallelGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_sequence import (
    SequenceParallelGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_tensor import (
    TensorParallelGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_tensor_1d import (
    TensorParallel1DGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_tensor_2d import (
    TensorParallel2DGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_tensor_2p5d import (
    TensorParallel2p5DGroupInitializer,
)
from oslo.torch.distributed._initializers.initializer_tensor_3d import (
    TensorParallel3DGroupInitializer,
)
from oslo.torch.distributed._parallel_mode import ParallelMode
from oslo.torch.distributed._seed.helper import add_seed, set_mode

TensorParallelGroupInitializerByMode = {
    None: None,
    "1d": TensorParallel1DGroupInitializer,
    "2d": TensorParallel2DGroupInitializer,
    "2.5d": TensorParallel2p5DGroupInitializer,
    "3d": TensorParallel3DGroupInitializer,
    "sequence": SequenceParallelGroupInitializer,
}


class ParallelContext(object):
    """
    Parallel Context class

    This class provides interface functions for users to get the parallel context,
    such as the global rank, the local rank, the world size, etc. of each device.

    Args:
        data_parallel_size (int): data parallel size
        pipeline_parallel_size (int): pipeline parallel size
        tensor_parallel_size (int): tensor parallel size
        tensor_parallel_mode (str): tensor parallel mode
        tensor_parallel_depth (int): tesseract depth for tensor 2.5 parallelism
        backend (str): distributed backend
        seed (int): random seed value

    Notes:
        Let's say we have a total of 16 GPUs denoted g0 ... g15 and we use 2 GPUs to parallelize the model tensor,
        and 4 GPUs to parallelize the model pipeline. The present method will create 8 model-parallel group,
        4 pipeline parallel groups and 8 data parallel groups as:

        - width: 4 pipeline parallel group
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        - height: 8 tensor parallel group
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        - depth: 8 data parallel group
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]

                        [g02, g06, g10, g14]
                      /  |              /  |
                     [g00, g04, g08, g12]  |
                     |   |             |   |
        3D parallel  |  [g03, g07, g11, g15]
                     |  /              |  /
                     [g01, g05, g09, g13]

                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
             tensor  | g00 |  |   g00    |  |   g04    |  |   g08    |  |   g12    |  | g12 |
        data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
             tensor  | g01 |  |   g01    |  |   g05    |  |   g09    |  |   g13    |  | g13 |
                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                    pipeline    pipeline      pipeline      pipeline      pipeline    pipeline

                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
             tensor  | g02 |  |   g02    |  |   g06    |  |   g10    |  |   g14    |  | g14 |
        data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
             tensor  | g03 |  |   g03    |  |   g07    |  |   g11    |  |   g15    |  | g15 |
                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                    pipeline    pipeline      pipeline      pipeline      pipeline    pipeline

    Examples:
        >>> from oslo.torch.distributed import ParallelContext

        >>> # Initialize from torch.distributed.launch
        >>> parallel_context = ParallelContext.from_torch(
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # Initialize from SLURM launcher
        >>> parallel_context = ParallelContext.from_slurm(
        ...     host="MY_HOST",
        ...     port=1234,
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # Initialize from OpenMPI launcher
        >>> parallel_context = ParallelContext.from_openmpi(
        ...     host="MY_HOST",
        ...     port=1234,
        ...     data_parallel_size=1,
        ...     pipeline_parallel_size=1,
        ...     tensor_parallel_size=1,
        ... )

        >>> # parallel_context world size
        >>> parallel_context.get_world_size(ParallelMode.DATA)

        >>> # get local size
        >>> parallel_context.get_local_rank(ParallelMode.DATA)

        >>> # get group
        >>> parallel_context.get_group(ParallelMode.DATA)

        >>> # get cpu group (gloo backend)
        >>> parallel_context.get_cpu_group(ParallelMode.DATA)

        >>> # get whole ranks in group
        >>> parallel_context.get_ranks_in_group(ParallelMode.DATA)

        >>> # get next global rank
        >>> parallel_context.get_next_global_rank(ParallelMode.DATA)

        >>> # get prev global rank
        >>> parallel_context.get_prev_global_rank(ParallelMode.DATA)
    """

    @classmethod
    def from_torch(
        cls,
        data_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        tensor_parallel_mode: Optional[str] = "1d",
        tensor_parallel_depth: Optional[int] = None,
        backend: str = "nccl",
        seed: bool = 42,
    ):
        """
        Initialize parallel context from `torch.distributed.launch`.

        Args:
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            tensor_parallel_mode (Optional[str]): tensor parallel mode
            tensor_parallel_depth (Optional[int]): tesseract depth for tensor 2.5 parallelism
            backend (str): distributed backend
            seed (int): random seed value

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from torch.distributed.launch
            >>> parallel_context = ParallelContext.from_torch(
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
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
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_depth: Optional[int] = None,
        backend: str = "nccl",
        seed: bool = 42,
        local_rank: Optional[int] = None,
    ):
        """
        Initialize parallel context from SLURM launcher.

        Args:
            host (str): host server
            port (int): communication port
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            tensor_parallel_mode (Optional[str]): tensor parallel mode
            tensor_parallel_depth (Optional[int]): tesseract depth for tensor 2.5 parallelism
            backend (str): distributed backend
            seed (int): random seed value
            local_rank (Optional[int]): local rank

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from SLURM launcher
            >>> parallel_context = ParallelContext.from_slurm(
            ...     host="MY_HOST",
            ...     port=1234,
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
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
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_depth: Optional[int] = None,
        backend: str = "nccl",
        seed: bool = 42,
    ):
        """
        Initialize parallel context from OpenMPI launcher.

        Args:
            host (str): host server
            port (int): communication port
            data_parallel_size (int): data parallel size
            pipeline_parallel_size (int): pipeline parallel size
            tensor_parallel_size (int): tensor parallel size
            tensor_parallel_mode (Optional[str]): tensor parallel mode
            tensor_parallel_depth (Optional[int]): tesseract depth for tensor 2.5 parallelism
            backend (str): distributed backend
            seed (int): random seed value

        Returns:
            ParallelContext: parallel context object

        Examples:
            >>> # Initialize from OpenMPI launcher
            >>> parallel_context = ParallelContext.from_openmpi(
            ...     host="MY_HOST",
            ...     port=1234,
            ...     data_parallel_size=1,
            ...     pipeline_parallel_size=1,
            ...     tensor_parallel_size=1,
            ... )
        """
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
        local_rank: Optional[int],
        world_size: int,
        host: str,
        port: int,
        data_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        tensor_parallel_mode: Optional[str],
        tensor_parallel_depth: Optional[int],
        backend: str,
        seed: int,
    ):
        assert tensor_parallel_mode in TensorParallelGroupInitializerByMode, (
            f"param `tensor_parallel_mode` {tensor_parallel_mode} is not available. "
            f"currently, we supports {list(TensorParallelGroupInitializerByMode.keys())}."
        )

        if tensor_parallel_size > 1:
            assert tensor_parallel_mode is not None, (
                "param `tensor_parallel_mode` must not be None "
                "if param `tensor_parallel_size` > 1."
            )

        if tensor_parallel_mode == "2.5d":
            assert tensor_parallel_depth is not None, (
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
        self._ranks_to_device = {}

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
        self._make_ranks_to_devices()

    # sanity check
    @staticmethod
    def _check_parallel_mode(parallel_mode: ParallelMode):
        """
        Check parallel_mode is ParallelMode object.

        Args:
            parallel_mode (ParallelMode): ParallelMode object
        """
        assert isinstance(parallel_mode, ParallelMode)

    # world sizes
    def get_world_size(self, parallel_mode: ParallelMode) -> int:
        """
        Get world size by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            int: world size by given parallel mode

        Examples:
            >>> parallel_context.get_world_size(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)
        return self._world_sizes[parallel_mode]

    def add_world_size(self, parallel_mode: ParallelMode, world_size: int):
        """
        Add world size for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object
            world_size (int): world size

        Examples:
            >>> parallel_context.add_world_size(ParallelMode.DATA, world_size=16)
        """
        self._check_parallel_mode(parallel_mode)
        self._world_sizes[parallel_mode] = world_size

    # local ranks
    def get_local_rank(self, parallel_mode: ParallelMode) -> int:
        """
        Get local rank by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            int: local rank by given parallel mode

        Examples:
            >>> parallel_context.get_local_rank(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)
        return self._local_ranks[parallel_mode]

    def add_local_rank(self, parallel_mode: ParallelMode, rank: int):
        """
        Add local rank for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object
            rank (int): world size

        Examples:
            >>> parallel_context.add_local_rank(ParallelMode.DATA, rank=4)
        """
        self._check_parallel_mode(parallel_mode)
        self._local_ranks[parallel_mode] = rank

    # global ranks
    def get_global_rank(self) -> int:
        """
        Get global rank

        Returns:
            int: global rank

        Examples:
            >>> parallel_context.get_global_rank()
        """
        return self._global_ranks[ParallelMode.GLOBAL]

    def add_global_rank(self, parallel_mode: ParallelMode, rank: int):
        """
        Add global rank for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object
            rank (int): world size

        Examples:
            >>> parallel_context.add_global_rank(ParallelMode.DATA, rank=4)
        """
        self._check_parallel_mode(parallel_mode)
        self._global_ranks[parallel_mode] = rank

    def get_next_global_rank(self, parallel_mode: ParallelMode) -> int:
        """
        Get next global rank by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            int: The next global rank by given parallel mode

        Examples:
            >>> parallel_context.get_next_global_rank(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)

        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank + 1) % world_size]

    def get_prev_global_rank(self, parallel_mode: ParallelMode) -> int:
        """
        Get previous global rank by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            int: The next global rank by given parallel mode

        Examples:
            >>> parallel_context.get_prev_global_rank(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)

        local_rank = self.get_local_rank(parallel_mode)
        world_size = self.get_world_size(parallel_mode)
        ranks_in_group = self.get_ranks_in_group(parallel_mode)

        return ranks_in_group[(local_rank - 1) % world_size]

    def is_first_rank(self, parallel_mode: ParallelMode) -> bool:
        """
        Is first rank in parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            bool: whether this rank is the first in given parallel mode

        Examples:
            >>> parallel_context.is_first_rank(ParallelMode.DATA)
        """
        return self.get_local_rank(parallel_mode) == 0

    def is_last_rank(self, parallel_mode):
        """
        Is last rank in parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            bool: whether this rank is the last in given parallel mode

        Examples:
            >>> parallel_context.is_last_rank(ParallelMode.DATA)
        """
        return (
            self.get_local_rank(parallel_mode) == self.get_world_size(parallel_mode) - 1
        )

    # groups
    def get_group(self, parallel_mode: ParallelMode) -> dist.ProcessGroup:
        """
        Get process group by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            dist.ProcessGroup: process group by given parallel mode

        Examples:
            >>> parallel_context.get_group(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)
        return self._groups[parallel_mode]

    def add_group(self, parallel_mode: ParallelMode, group: dist.ProcessGroup):
        """
        Add process group for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object
            group (dist.ProcessGroup): process group

        Examples:
            >>> parallel_context.add_global_rank(ParallelMode.DATA, rank=4)
        """
        self._check_parallel_mode(parallel_mode)
        self._groups[parallel_mode] = group

    # cpu groups
    def get_cpu_group(self, parallel_mode: ParallelMode):
        """
        Get CPU process group by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            dist.ProcessGroup: process group by given parallel mode

        Notes:
            this is process group using gloo backend

        Examples:
            >>> parallel_context.get_group(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)
        return self._cpu_groups[parallel_mode]

    def add_cpu_group(self, parallel_mode, group: ParallelMode):
        """
        Add CPU process group for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Notes:
            this is process group using gloo backend

        Examples:
            >>> parallel_context.add_cpu_group(ParallelMode.DATA, group=MY_GROUP)
        """
        self._check_parallel_mode(parallel_mode)
        self._cpu_groups[parallel_mode] = group

    # ranks in group
    def get_ranks_in_group(self, parallel_mode: ParallelMode) -> List[int]:
        """
        Get whole ranks in the group by given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            List[int]: Whole ranks in the group by given parallel mode

        Examples:
            >>> parallel_context.get_ranks_in_group(ParallelMode.DATA)
        """
        self._check_parallel_mode(parallel_mode)
        return self._ranks_in_group[parallel_mode]

    def add_ranks_in_group(self, parallel_mode: ParallelMode, ranks: List[int]):
        """
        Add whole ranks in the group for given parallel mode

        Args:
            parallel_mode (ParallelMode): ParallelMode object
            ranks (List[int]): ranks in group

        Examples:
            >>> parallel_context.add_ranks_in_group(ParallelMode.DATA, ranks=[0, 2, 5, 8])
        """
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
        """
        Initialize global distributed process group

        Args:
            rank (int): global rank
            world_size (int): global world size
            backend (int): distributed backend
            host (str): host server
            port (int): communication port
        """
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
        group_world_size: int,
        process_group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        ranks_in_group: List[int],
        mode: ParallelMode,
    ):
        """
        Register distributed setting by give parallel mode

        Args:
            local_rank (int): local rank
            group_world_size (int): group world size
            process_group (dist.ProcessGroup): process group
            cpu_group (dist.ProcessGroup): cpu process group
            ranks_in_group (List[int]): whole ranks in the group
            mode (ParallelMode): ParallelMode object
        """
        self.add_local_rank(mode, local_rank)
        self.add_world_size(mode, group_world_size)
        self.add_group(mode, process_group)
        self.add_cpu_group(mode, cpu_group)
        self.add_ranks_in_group(mode, ranks_in_group)

    def init_parallel_groups(self):
        """Initialize whole parallel groups"""
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

    def _make_ranks_to_devices(self):
        rank_tensor = torch.zeros(len(self._local_ranks), dtype=torch.long).cuda()

        for idx, local_rank in enumerate(self._local_ranks.values()):
            rank_tensor[idx] = local_rank

        rank_tensor_list = [
            torch.zeros(rank_tensor.size(), dtype=torch.long).cuda()
            for _ in range(self.get_world_size(ParallelMode.GLOBAL))
        ]

        dist.all_gather(tensor_list=rank_tensor_list, tensor=rank_tensor)

        for _rank, _rank_tensor in enumerate(rank_tensor_list):
            modes_and_ranks = {
                mode: rank
                for mode, rank in zip(self._local_ranks.keys(), _rank_tensor.tolist())
            }
            self._ranks_to_device[tuple(modes_and_ranks.items())] = _rank

    def ranks2device(self, ranks):
        """
        Examples:
            ranks:
                {
                    <ParallelMode.TENSOR_1D: 'tensor_1d'>: 1
                    <ParallelMode.DATA: 'data'>: 0
                }

            self._ranks_to_device:
            {
                (
                    (<ParallelMode.GLOBAL: 'global'>, 0),
                    (<ParallelMode.DATA: 'data'>, 0),
                    (<ParallelMode.MODEL: 'model'>, 0),
                    (<ParallelMode.TENSOR: 'tensor'>, 0),
                    (<ParallelMode.TENSOR_1D: 'tensor_1d'>, 0)
                ): 0,
                (
                    (<ParallelMode.GLOBAL: 'global'>, 1),
                    (<ParallelMode.DATA: 'data'>, 0),
                    (<ParallelMode.MODEL: 'model'>, 1),
                    (<ParallelMode.TENSOR: 'tensor'>, 1),
                    (<ParallelMode.TENSOR_1D: 'tensor_1d'>, 1)
                ): 1,
                ...
            }

            return device: 1
        """
        ranks_key = {mode: None for mode in self._local_ranks.keys()}

        for mode in self._local_ranks.keys():
            if mode in ranks:
                ranks_key[mode] = ranks[mode]
            else:
                ranks_key[mode] = self.get_local_rank(mode)

        return self._ranks_to_device[tuple(ranks_key.items())]

    def is_initialized(self, parallel_mode: ParallelMode):
        """
        Check whether it's initialized or not.

        Args:
            parallel_mode (ParallelMode): ParallelMode object

        Returns:
            bool: Whether it's initialized or not.
        """
        return parallel_mode in self._groups

    def destroy(self):
        """Destroy all the parallel groups"""
        for mode, group in self._groups.items():
            if mode is not ParallelMode.GLOBAL:
                dist.destroy_process_group(group)

        dist.destroy_process_group()
        self._groups.clear()

    def set_device(self, device_ordinal: Optional[int] = None):
        """
        Set CUDA device

        Args:
            device_ordinal (Optional[int]): device ordinal
        """
        global_rank = self.get_global_rank()
        if device_ordinal is None:
            devices_per_node = torch.cuda.device_count()
            device_ordinal = global_rank % devices_per_node
        torch.cuda.set_device(device_ordinal)

    def set_seed(self, seed: int):
        """
        Set seed value

        Args:
            seed (int): seed value
        """
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
