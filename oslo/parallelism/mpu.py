# Copyright 2021 TUNiB Inc.

import os
from math import floor
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor, Size
from torch.autograd import Function

from oslo.parallelism.utils import NoneType


class LayerInfo:
    @staticmethod
    def base():
        """
        Returns:
            the base transformer block of model

        Examples:
            >>> return BertLayer
        """
        raise NotImplementedError

    @staticmethod
    def attention():
        """
        Returns:
            the last elements of attention modules

        Examples:
            >>> return BertAttention
        """
        raise NotImplementedError

    @staticmethod
    def mlp():
        """
        Returns:
            the last element of mlp modules

        Examples:
            >>> return BertOutput
        """
        raise NotImplementedError

    @staticmethod
    def reducing_required():
        """
        Returns:
            arguments that are required reducing

        Examples:
            >>> return ["all_head_size", "num_attention_heads"]
        """
        raise NotImplementedError


class MPU(object):
    """
    MPU: Model Parallel Unit

    MPU is key concept of Oslo framework and is inspired by Megatron-LM.
    The main difference with Megatron-LM is that each model has an their mpu.

    We can combine several models later. For example, in the case of Electra,
    there is a generator model and a discriminator model. To parallelize all of them,
    each model must be parallelized in a different process group,
    so the mpu must be maintained in the model level, not the global state.

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
              model  | g00 |  |   g00    |  |   g04    |  |   g08    |  |   g12    |  | g12 |
        data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
              model  | g01 |  |   g01    |  |   g05    |  |   g09    |  |   g13    |  | g13 |
                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                    embedding   pipeline      pipeline      pipeline      pipeline   embedding

                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
              model  | g02 |  |   g02    |  |   g06    |  |   g10    |  |   g14    |  | g14 |
        data         +-----+  +----------+  +----------+  +----------+  +----------+  +-----+  ===> forward
              model  | g03 |  |   g03    |  |   g07    |  |   g11    |  |   g15    |  | g15 |
                     +-----+  +----------+  +----------+  +----------+  +----------+  +-----+
                    embedding   pipeline      pipeline      pipeline      pipeline   embedding

    References:
        Original MPU implementation of Megatron-LM. We refactored all the code to be pythonic.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/initialize.py

    """

    # process group
    _data_parallel_group = None
    _model_parallel_group = None
    _tensor_parallel_group = None
    _pipeline_parallel_group = None
    _p2p_communication_groups = None
    _embedding_tied_group = None

    # world size
    _data_parallel_world_size = None
    _model_parallel_world_size = None
    _tensor_parallel_world_size = None
    _pipeline_parallel_world_size = None
    _p2p_communication_world_size = 2
    _embedding_tied_world_size = 2

    # single rank
    _data_parallel_rank = None
    _model_parallel_rank = None
    _tensor_parallel_rank = None
    _pipeline_parallel_rank = None

    # global ranks
    _pipeline_parallel_global_ranks_per_group = None
    _pipeline_parallel_global_ranks_across_all_groups = []
    _p2p_communication_global_ranks = None
    _embedding_tied_global_ranks = None

    def __init__(self, tensor_parallel_size: int, pipeline_parallel_size: int) -> None:
        """
        Initialize MPU object.
        All the process groups are initialized when this method is invoked.

        Args:
            tensor_parallel_size (int): model parallel world size
            pipeline_parallel_size (int): pipeline parallel world size
            master_port (int): master port to initialize global process group
        """

        if not dist.is_initialized():
            self.initialize_distributed()

        current_rank = dist.get_rank()
        global_world_size = dist.get_world_size()

        assert (
            global_world_size >= tensor_parallel_size
        ), "param `tensor_parallel_size` must be smaller than global world size."

        assert (
            global_world_size >= pipeline_parallel_size
        ), "param `pipeline_model_parallel_size` must be smaller than global world size."

        total_model_parallel_size = tensor_parallel_size * pipeline_parallel_size

        assert (
            global_world_size % total_model_parallel_size == 0
        ), "global world sizes must be divisible by model parallel world sizes (tp * pp)"

        num_tensor_parallel_groups = global_world_size // tensor_parallel_size
        num_pipeline_parallel_groups = global_world_size // pipeline_parallel_size

        # 1. initialize data parallel group
        all_data_parallel_group_ranks = self._initialize_data_parallel_group(
            current_rank=current_rank,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            num_pipeline_parallel_groups=num_pipeline_parallel_groups,
        )

        self._initialize_model_parallel_groups(
            current_rank=current_rank,
            all_data_parallel_group_ranks=all_data_parallel_group_ranks,
        )

        # 3. initialize tensor model parallel group
        self._initialize_tensor_parallel_group(
            current_rank=current_rank,
            tensor_parallel_size=tensor_parallel_size,
            num_tensor_parallel_groups=num_tensor_parallel_groups,
        )

        # 4. initialize pipeline model parallel group
        self._initialize_pipeline_parallel_group(
            current_rank=current_rank,
            global_world_size=global_world_size,
            num_pipeline_parallel_groups=num_pipeline_parallel_groups,
        )

        if pipeline_parallel_size > 1:
            # 5. initialize p2p communication group
            self._initialize_p2p_communication_group(
                global_world_size=global_world_size,
            )

        # 6. create distributed functions
        functions = self._initialize_functions()
        self._broadcast_fn = functions["broadcast"]
        self._reduce_fn = functions["reduce"]
        self._scatter_fn = functions["scatter"]
        self._gather_fn = functions["gather"]

    # Initialization

    @staticmethod
    def initialize_distributed():
        """Initialize torch.distributed and mpu."""
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        if not dist.is_initialized():
            device_count = torch.cuda.device_count()

            if device_count > 0:
                device = rank % device_count
                torch.cuda.set_device(device)

            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", 29500)
            init_method += str(master_ip) + ":" + str(master_port)

            torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

    def _initialize_data_parallel_group(
        self,
        current_rank: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        num_pipeline_parallel_groups: int,
    ) -> List[List[int]]:
        """
        Initialize data parallel group

        Args:
            current_rank (int): current rank
            tensor_parallel_size (int): tensor model parallel world size
            pipeline_parallel_size (int): pipeline model parallel world size
            num_pipeline_parallel_groups (int): the number of pipeline model parallel groups

        Returns:
            List[List[int]]: all_data_parallel_group_ranks
        """
        assert (
            self._data_parallel_group is None
        ), "data parallel group is already initialized."

        all_data_parallel_group_ranks = []
        for i in range(pipeline_parallel_size):
            start_rank = i * num_pipeline_parallel_groups
            end_rank = (i + 1) * num_pipeline_parallel_groups

            for j in range(tensor_parallel_size):
                ranks = list(range(start_rank + j, end_rank, tensor_parallel_size))
                all_data_parallel_group_ranks.append(ranks)
                group = dist.new_group(ranks)
                if current_rank in ranks:
                    self._data_parallel_group = group

        return all_data_parallel_group_ranks

    def _initialize_model_parallel_groups(
        self,
        current_rank: int,
        all_data_parallel_group_ranks: List[List[int]],
    ):
        # Build the model-parallel groups.
        assert (
            self._model_parallel_group is None
        ), "model parallel group is already initialized"
        for i in range(self.get_data_parallel_world_size()):
            ranks = [
                data_parallel_group_ranks[i]
                for data_parallel_group_ranks in all_data_parallel_group_ranks
            ]
            group = torch.distributed.new_group(ranks)
            if current_rank in ranks:
                self._model_parallel_group = group

    def _initialize_tensor_parallel_group(
        self,
        current_rank: int,
        tensor_parallel_size: int,
        num_tensor_parallel_groups: int,
    ) -> None:
        """
        Initialize tensor model parallel group

        Args:
            current_rank (int): current rank
            tensor_parallel_size (int): tensor model parallel world size
            num_tensor_parallel_groups (int): the number of tensor model parallel groups
        """
        assert (
            self._tensor_parallel_group is None
        ), "tensor model parallel group is already initialized."

        for i in range(num_tensor_parallel_groups):
            start_rank = i * tensor_parallel_size
            end_rank = (i + 1) * tensor_parallel_size

            ranks = list(range(start_rank, end_rank))
            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._tensor_parallel_group = group

    def _initialize_pipeline_parallel_group(
        self,
        current_rank: int,
        global_world_size: int,
        num_pipeline_parallel_groups: int,
    ) -> None:
        """
        Initialize pipeline model parallel group

        Args:
            current_rank (int): current rank
            global_world_size (int): global world size
            num_pipeline_parallel_groups (int): the number of tensor model parallel groups
        """
        assert (
            self._pipeline_parallel_group is None
        ), "pipeline model parallel group is already initialized."

        for i in range(num_pipeline_parallel_groups):
            ranks = list(range(i, global_world_size, num_pipeline_parallel_groups))
            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._pipeline_parallel_group = group
                self._pipeline_model_parallel_global_ranks_per_group = ranks

            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
            else:
                embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)

            if current_rank in embedding_ranks:
                self._embedding_tied_group = group
            if current_rank in ranks:
                self._embedding_tied_global_ranks = embedding_ranks

            self._pipeline_parallel_global_ranks_across_all_groups.append(ranks)

    def _initialize_p2p_communication_group(self, global_world_size: int) -> None:
        """
        Initialize P2P communication groups

        Args:
            global_world_size (int): global world size
        """
        assert (
            self._p2p_communication_groups is None
        ), "pipeline model parallel group is already initialized."

        ranks = []

        for rank in range(global_world_size):
            for l in self.get_pipeline_parallel_global_ranks_across_all_groups():
                if rank in l:
                    idx = l.index(rank)
                    buddy_rank = l[(idx + 1) % self.get_pipeline_parallel_world_size()]
                    ranks.append([rank, buddy_rank])
                    break  # next global rank

        assert len(ranks) == global_world_size
        self._p2p_communication_global_ranks = ranks
        send_groups = {(r[0], r[1]): dist.new_group(r) for r in ranks}
        recv_groups = {(r[1], r[0]): dist.new_group(r) for r in ranks}
        self_groups = {(r,): dist.new_group([r]) for r in range(global_world_size)}

        self._p2p_communication_groups = {}
        self._p2p_communication_groups.update(send_groups)
        self._p2p_communication_groups.update(recv_groups)
        self._p2p_communication_groups.update(self_groups)

    def model_parallel_is_initialized(self) -> bool:
        """
        Check if model parallel groups are initialized.

        Returns:
            bool: whether model parallel groups are initialized
        """
        if (
            self._tensor_parallel_group is None
            or self._pipeline_parallel_group is None
            or self._data_parallel_group is None
        ):
            return False
        return True

    # Process group

    def get_data_parallel_group(self):
        """
        Get the data parallel group.

        Returns:
            ProcessGroup: data parallel group
        """
        assert (
            self._data_parallel_group is not None
        ), "data parallel group is not initialized."

        return self._data_parallel_group

    def get_model_parallel_group(self):
        """
        Get the model parallel group.

        Returns:
            ProcessGroup: model parallel group
        """
        assert (
            self._model_parallel_group is not None
        ), "model parallel group is not initialized."

        return self._model_parallel_group

    def get_tensor_parallel_group(self):
        """
        Get the tensor model parallel group.

        Returns:
            ProcessGroup: tensor model parallel group
        """
        assert (
            self._tensor_parallel_group is not None
        ), "tensor model parallel group is not initialized."

        return self._tensor_parallel_group

    def get_pipeline_parallel_group(self):
        """
        Get the pipeline model parallel group.

        Returns:
            ProcessGroup: pipeline model parallel group
        """
        assert (
            self._pipeline_parallel_group is not None
        ), "pipeline model parallel group is not initialized."

        return self._pipeline_parallel_group

    def get_p2p_communication_groups(self):
        """
        Get the p2p communication groups.

        Returns:
            ProcessGroup: p2p communication groups
        """
        assert (
            self._p2p_communication_groups is not None
        ), "p2p communication groups are not initialized."

        return self._p2p_communication_groups

    def get_embedding_tied_group(self):
        """
        Get the embedding tied group.

        Returns:
            ProcessGroup: embedding tied group
        """
        assert (
            self._embedding_tied_group is not None
        ), "embedding tied group is not initialized."

        return self._embedding_tied_group

    # World Size

    def get_data_parallel_world_size(self) -> int:
        """
        Get the data parallel world size

        Returns:
            int: data parallel world size
        """
        if self._data_parallel_world_size is not None:
            return self._data_parallel_world_size

        return dist.get_world_size(self.get_data_parallel_group())

    def get_model_parallel_world_size(self):
        """
        Get the model parallel world size

        Returns:
            int: model parallel world size
        """
        if self._model_parallel_world_size is not None:
            return self._model_parallel_world_size

        return dist.get_world_size(self.get_model_parallel_group())

    def get_tensor_parallel_world_size(self) -> int:
        """
        Get the tensor model parallel world size

        Returns:
            int: tensor model parallel world size
        """
        if self._tensor_parallel_world_size is not None:
            return self._tensor_parallel_world_size

        return dist.get_world_size(self.get_tensor_parallel_group())

    def get_pipeline_parallel_world_size(self) -> int:
        """
        Get the pipeline model parallel world size

        Returns:
            int: pipeline model parallel world size
        """
        if self._pipeline_parallel_world_size is not None:
            return self._pipeline_parallel_world_size

        return dist.get_world_size(self.get_pipeline_parallel_group())

    def get_p2p_communication_world_size(self) -> int:
        """
        Get the p2p communication world size

        Returns:
            int: p2p communication world size
        """
        return self._p2p_communication_world_size

    def get_embedding_tied_world_size(self) -> int:
        """
        Get the embedding tied world size

        Returns:
            int: embedding tied world size
        """
        return self._embedding_tied_world_size

    # Single Rank

    def get_data_parallel_rank(self) -> int:
        """
        Get the data parallel rank

        Returns:
            int: data parallel rank
        """
        if self._data_parallel_rank is not None:
            return self._data_parallel_rank

        return dist.get_rank(self.get_data_parallel_group())

    def get_model_parallel_rank(self) -> int:
        """
        Get the model parallel rank

        Returns:
            int: model parallel rank
        """
        if self._model_parallel_rank is not None:
            return self._model_parallel_rank

        return dist.get_rank(self.get_model_parallel_group())

    def get_tensor_parallel_rank(self) -> int:
        """
        Get the tensor model parallel rank

        Returns:
            int: tensor model parallel world size
        """
        if self._tensor_parallel_rank is not None:
            return self._tensor_parallel_rank

        return dist.get_rank(self.get_tensor_parallel_group())

    def get_pipeline_parallel_rank(self) -> int:
        """
        Get pipeline model parallel rank

        Returns:
            int: pipeline model parallel rank
        """
        if self._pipeline_parallel_rank is not None:
            return self._pipeline_parallel_rank

        return dist.get_rank(self.get_pipeline_parallel_group())

    def get_tensor_parallel_src_rank(self) -> int:
        """
        Compute the global rank corresponding to the first local rank in the tensor model parallel group.

        Returns:
            int: tensor model parallel source rank
        """
        global_rank = dist.get_rank()
        local_world_size = self.get_tensor_parallel_world_size()
        return (global_rank // local_world_size) * local_world_size

    def is_pipeline_first_stage(self) -> bool:
        """
        Return `True` if in the first pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is first
        """
        return self.get_pipeline_parallel_rank() == 0

    def is_pipeline_last_stage(self) -> bool:
        """
        Return `True` if in the last pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is last
        """
        return self.get_pipeline_parallel_rank() == (
            self.get_pipeline_parallel_world_size() - 1
        )

    # Global Ranks

    def get_pipeline_parallel_global_ranks_across_all_groups(self):
        """
        Get all the pipeline model parallel ranks across all groups

        Returns:
            List[List[int]]: all the parallel ranks across all devices
        """
        return self._pipeline_parallel_global_ranks_across_all_groups

    def get_pipeline_model_parallel_global_ranks_per_group(self):
        """
        Get all the pipeline model parallel ranks per single group

        Returns:
            List[int]: all the parallel ranks
        """
        return self._pipeline_model_parallel_global_ranks_per_group

    def get_p2p_communication_global_ranks(self):
        assert (
            self._p2p_communication_groups is not None
        ), "p2p communication groups are not initialized."

        return self._p2p_communication_global_ranks

    def get_embedding_global_ranks(self):
        """
        Get all the embedding ranks

        Returns:
            List[int]: all the parallel ranks
        """
        return self._embedding_tied_global_ranks

    # Utils functions

    def destroy_model_parallel(self) -> None:
        """
        Destroy all the model parallel groups
        """

        self._tensor_parallel_group = None
        self._pipeline_parallel_group = None
        self._data_parallel_group = None
        self._p2p_communication_groups = None
        self._embedding_tied_group = None

    def _broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the tensor model parallel region.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: broadcast tensor
        """

        if not inputs.is_cuda:
            inputs = inputs.cuda()

        return inputs

    def _reduce(self, inputs: Tensor):
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """
        if self.get_tensor_parallel_world_size() == 1:
            return inputs

        dist.all_reduce(
            tensor=inputs,
            group=self.get_tensor_parallel_group(),
        )

        return inputs

    def _scatter(self, inputs: Tensor, dim: int = -1) -> Tensor:
        """
        Split the tensor along given dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor
            dim (int): scatter dimension

        Returns:
            Tensor: scattered tensor
        """
        world_size = self.get_tensor_parallel_world_size()

        if world_size == 1:
            return inputs

        inputs_list = torch.chunk(input=inputs, chunks=world_size, dim=dim)
        # In megatron-lm, `torch.split` is used,
        # but `torch.chunk` is correct to scatter odd-sized tensors.

        rank = self.get_tensor_parallel_rank()
        outputs = inputs_list[rank].contiguous()
        return outputs

    def _gather(self, inputs: Tensor, dim: int = -1) -> Tensor:
        """
        Gather tensors and concatenate along the given dimension
        We support different size tensor all-gather.

        Args:
            inputs (Tensor): input tensor
            dim (int): gather dimension

        Returns:
            Tensor: gathered tensor
        """

        world_size = self.get_tensor_parallel_world_size()

        if world_size == 1:
            return inputs

        input_size = list(inputs.size())

        size_tensor = torch.tensor(
            input_size[dim], device=inputs.device, dtype=torch.long
        )
        sizes_list = [torch.empty_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(sizes_list, size_tensor, group=self.get_tensor_parallel_group())

        sizes_list = [_.item() for _ in sizes_list]
        max_size = max(sizes_list)
        pad_size = max_size - input_size[dim]

        if pad_size > 0:
            pad_tensor_size = list(inputs.size())
            pad_tensor_size[dim] = pad_size
            pad = torch.zeros(
                *pad_tensor_size,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            inputs = torch.cat([inputs, pad], dim=dim)

        tensor_list = [
            torch.empty(
                *inputs.size(),
                device=inputs.device,
                dtype=inputs.dtype,
            )
            for _ in range(world_size)
        ]

        if torch.is_tensor(inputs):
            if not inputs.is_contiguous():
                inputs = inputs.contiguous()

        dist.all_gather(tensor_list, inputs, group=self.get_tensor_parallel_group())

        # remove padding ;)
        for i, orig_size in enumerate(sizes_list):
            tensor_list[i] = torch.split(tensor_list[i], orig_size, dim=dim)[0]

        return torch.cat(tensor_list, dim=dim)

    def broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the tensor model parallel region.

        Args:
            inputs (Tensor):

        Returns:
            Tensor: broadcast tensor
        """

        if self._enable_grad(inputs):
            outputs = self._broadcast_fn.apply(inputs)
        else:
            outputs = self._broadcast(inputs)
        return outputs

    def reduce(self, inputs: Tensor) -> Tensor:
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """

        if self._enable_grad(inputs):
            outputs = self._reduce_fn.apply(inputs)
        else:
            outputs = self._reduce(inputs)
        return outputs

    def scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._scatter_fn.apply(inputs)
        else:
            outputs = self._scatter(inputs)
        return outputs

    def gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._gather_fn.apply(inputs)
        else:
            outputs = self._gather(inputs)
        return outputs

    @staticmethod
    def _enable_grad(inputs: Tensor) -> bool:
        """
        Check current tensor is enabled to pass gradient.

        Args:
            inputs (Tensor): input tensor

        Returns:
            bool: whether gradient can be passed or not
        """
        return torch.is_grad_enabled() and inputs.requires_grad

    def _initialize_functions(self):
        class Broadcast(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._broadcast(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._reduce(inputs)

        class Reduce(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._reduce(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._broadcast(inputs)

        class Scatter(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._scatter(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._gather(inputs)

        class Gather(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._gather(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._scatter(inputs)

        return {
            "broadcast": Broadcast,
            "reduce": Reduce,
            "scatter": Scatter,
            "gather": Gather,
        }

    def p2p(self, tensor: Tensor, src_stage: int, dst_stage: int):
        """
        P2P communication (send or receive) tensor using broadcast operation.

        Notes:
            NCCL only supports send and recv operation for torch 1.8+.
            But there are many users that uses torch under 1.8 version.
            So we implemented p2p communication using broadcast operation.

        Args:
            tensor (Tensor): input tensor
            src_stage (int): source pipeline stage
            dst_stage (int): destination pipeline stage
        """

        self._is_valid_send_recv(src_stage, dst_stage)
        stage_to_rank = self.get_pipeline_model_parallel_global_ranks_per_group()

        src_rank = stage_to_rank[src_stage]
        dst_rank = stage_to_rank[dst_stage]

        group_key = (src_rank, dst_rank)
        groups = self.get_p2p_communication_groups()

        if group_key in groups:
            dist.broadcast(tensor, src=src_rank, group=groups[group_key])
            return True

        return False

    def _is_valid_send_recv(self, src_stage, dst_stage):
        first_stage = 0
        last_stage = self.get_pipeline_parallel_world_size() - 1

        assert (
            abs(src_stage - dst_stage) in [0, 1]
            or (first_stage <= src_stage <= last_stage)
            or (first_stage <= dst_stage <= last_stage)
            or (src_stage == first_stage and dst_stage == last_stage)
            or (src_stage == last_stage and dst_stage == first_stage)
        ), f"Wrong stages: src={src_stage}, dst={dst_stage}\n"

    @staticmethod
    def make_pipeline_partition(num_items, num_parts):
        """compute range of pipeline partition for current rank"""
        parts = [0] * (num_parts + 1)
        # First check for the trivial edge case
        if num_items <= num_parts:
            for p in range(num_parts + 1):
                parts[p] = min(p, num_items)
            return parts

        chunksize = floor(num_items / num_parts)
        for p in range(num_parts):
            parts[p] = min(chunksize * p, num_items)
        parts[num_parts] = num_items
        return parts

    @staticmethod
    def extend_pipeline_partition(partition):
        last_part = len(partition) - 1
        ext_partition = []

        for i in range(len(partition)):
            if i != last_part:
                num_layers = partition[i + 1] - partition[i]
                ext_partition += [i] * num_layers

        return ext_partition


class P2PCommunicator(object):
    """
    P2P communicator that can communicate various data types.

    Args:
        mpu (MPU): model parallel unit
    """

    def __init__(self, mpu: MPU):
        self.mpu = mpu
        self.device = torch.device(torch.cuda.current_device())
        self.num_stages = mpu.get_pipeline_parallel_world_size()
        self.stage_id = mpu.get_pipeline_parallel_rank()
        self.INSTRUCTIONS = {
            bool: {"send": self.send_bool, "recv": self.recv_bool},
            int: {"send": self.send_int, "recv": self.recv_int},
            float: {"send": self.send_float, "recv": self.recv_float},
            complex: {"send": self.send_complex, "recv": self.recv_complex},
            str: {"send": self.send_str, "recv": self.recv_str},
            type: {"send": self.send_type, "recv": self.recv_type},
            list: {"send": self.send_list, "recv": self.recv_list},
            tuple: {"send": self.send_tuple, "recv": self.recv_tuple},
            set: {"send": self.send_set, "recv": self.recv_set},
            dict: {"send": self.send_dict, "recv": self.recv_dict},
            NoneType: {"send": self.send_none, "recv": self.recv_none},
            Size: {"send": self.send_size, "recv": self.recv_size},
            Tensor: {"send": self.send_tensor, "recv": self.recv_tensor},
        }

        self.TORCH_ID_TO_DTYPE = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]

        self.ID_TO_DTYPE = list(self.INSTRUCTIONS.keys())
        self.DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(self.ID_TO_DTYPE)}
        self.TORCH_DTYPE_TO_ID = {
            dtype: idx for idx, dtype in enumerate(self.TORCH_ID_TO_DTYPE)
        }

    def ring_exchange_forward(self, value):
        """
        Exchange values lower to higher stage

        e.g. 0 -> 1 -> 2 -> 3
        """
        if not self.mpu.is_pipeline_first_stage():
            value = self.recv(self.stage_id - 1)

        if not self.mpu.is_pipeline_last_stage():
            self.send(value, self.stage_id + 1)

        return value

    def ring_exchange_backward(self, value):
        """
        Exchange values higher to lower stage

        e.g. 3 -> 2 -> 1 -> 0
        """
        if not self.mpu.is_pipeline_last_stage():
            value = self.recv(self.stage_id + 1)

        if not self.mpu.is_pipeline_first_stage():
            self.send(value, self.stage_id - 1)

        return value

    def send(self, value, recv_stage):
        """Send data that have every type to other stage"""
        _type = type(value)
        assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"

        return self.INSTRUCTIONS[_type]["send"](
            value, recv_stage=recv_stage, send_type=True
        )

    def recv(self, send_stage):
        """Receive data that have every type from other stage"""
        _type = self.INSTRUCTIONS[type]["recv"](send_stage)
        return self.INSTRUCTIONS[_type]["recv"](send_stage)

    def send_type(self, _type, recv_stage, send_type=False):
        """Send type to other stage"""
        assert send_type is False, "to send ``type``, we don't need to send type."

        send_type = torch.tensor(
            [self.DTYPE_TO_ID[_type]],
            dtype=torch.long,
            device=self.device,
        )

        assert self.mpu.p2p(
            send_type, self.stage_id, recv_stage
        ), f"Communication failed: send_type_{self.stage_id}_to_{recv_stage}"

    def recv_type(self, send_stage):
        """Receive type from other stage"""
        recv_type = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_type, send_stage, self.stage_id
        ), f"Communication failed: recv_type_{send_stage}_to_{self.stage_id}"
        return self.ID_TO_DTYPE[recv_type.item()]

    def send_none(self, none, recv_stage, send_type=False):
        """Send None to other stage"""
        assert none is None, f"wrong type: {none} must be {NoneType} type"

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](NoneType, recv_stage)

    def recv_none(self, send_stage):
        """Receive None from other stage"""
        return None

    def send_str(self, _str, recv_stage, send_type=False):
        """Send string to other stage, note we only support ascii string now"""
        assert isinstance(_str, str), f"wrong type: {_str} must be {str} type."
        assert all(ord(c) < 128 for c in _str), "send string must be ascii string."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](str, recv_stage)

        send_len_string = torch.tensor(
            [len(_str)],
            dtype=torch.long,
            device=self.device,
        )

        assert self.mpu.p2p(
            send_len_string, self.stage_id, recv_stage
        ), f"Communication failed: send_str_{self.stage_id}_to_{recv_stage}"

        send_string = torch.tensor(
            [ord(s) for s in _str], dtype=torch.long, device=self.device
        )

        assert self.mpu.p2p(send_string, self.stage_id, recv_stage)

    def recv_str(self, send_stage):
        """Receive string from other stage"""
        recv_len_string = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(recv_len_string, send_stage, self.stage_id)
        recv_len_string = recv_len_string.item()

        recv_string = torch.tensor(
            [0] * recv_len_string,
            dtype=torch.long,
            device=self.device,
        )
        assert self.mpu.p2p(
            recv_string, send_stage, self.stage_id
        ), f"Communication failed: recv_type_{send_stage}_to_{self.stage_id}"
        return "".join([chr(s) for s in recv_string])

    def send_bool(self, _bool, recv_stage, send_type=False):
        """Send boolean to other stage"""
        assert isinstance(_bool, bool), f"wrong type: {_bool} must be {bool} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](bool, recv_stage)

        send_boolean = torch.tensor(
            [1 if _bool else 0], dtype=torch.long, device=self.device
        )

        assert self.mpu.p2p(
            send_boolean, self.stage_id, recv_stage
        ), f"Communication failed: send_bool_{self.stage_id}_to_{recv_stage}"

    def recv_bool(self, send_stage):
        """Receive boolean from other stage"""
        recv_boolean = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_boolean, send_stage, self.stage_id
        ), f"Communication failed: recv_bool_{send_stage}_to_{self.stage_id}"
        recv_boolean = recv_boolean.item()

        if recv_boolean == 0:
            return False
        elif recv_boolean == 1:
            return True
        else:
            raise ValueError(
                f"Wrong value for boolean. only 0 or 1 can be supported. "
                f"but your input is {recv_boolean}"
            )

    def send_int(self, _int, recv_stage, send_type=False):
        """Send int to other stage"""
        assert isinstance(_int, int), f"wrong type: {_int} must be {int} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](int, recv_stage)

        send_int = torch.tensor([_int], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            send_int, self.stage_id, recv_stage
        ), f"Communication failed: send_int_{self.stage_id}_to_{recv_stage}"

    def recv_int(self, send_stage):
        """Receive int from other stage"""
        recv_int = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_int, send_stage, self.stage_id
        ), f"Communication failed: recv_int_{send_stage}_to_{self.stage_id}"
        return recv_int.item()

    def send_float(self, _float, recv_stage, send_type=False):
        """Send float to other stage"""
        assert isinstance(_float, float), f"wrong type: {_float} must be {float} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](float, recv_stage)

        send_float = torch.tensor([_float], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_float, self.stage_id, recv_stage
        ), f"Communication failed: send_float_{self.stage_id}_to_{recv_stage}"

    def recv_float(self, send_stage):
        """Receive float from other stage"""
        recv_float = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_float, send_stage, self.stage_id
        ), f"Communication failed: recv_float_{send_stage}_to_{self.stage_id}"
        return recv_float.item()

    def send_complex(self, _complex, recv_stage, send_type=False):
        """Send complex to other stage"""
        assert isinstance(
            _complex, complex
        ), f"wrong type: {_complex} must be {complex} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](complex, recv_stage)

        send_real = torch.tensor([_complex.real], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_real, self.stage_id, recv_stage
        ), f"Communication failed: send_complex_{self.stage_id}_to_{recv_stage}"
        send_imag = torch.tensor([_complex.imag], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_imag, self.stage_id, recv_stage
        ), f"Communication failed: send_complex_{self.stage_id}_to_{recv_stage}"

    def recv_complex(self, send_stage):
        """Receive complex from other stage"""
        recv_real = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_real, send_stage, self.stage_id
        ), f"Communication failed: recv_complex_{send_stage}_to_{self.stage_id}"
        recv_imag = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_imag, send_stage, self.stage_id
        ), f"Communication failed: recv_complex_{send_stage}_to_{self.stage_id}"

        return complex(recv_real.item(), recv_imag.item())

    def send_tensor(self, _tensor, recv_stage, send_type=False):
        """Send tensor to other stage"""
        assert isinstance(
            _tensor, Tensor
        ), f"wrong type: {_tensor} must be {Tensor} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](Tensor, recv_stage)

        # dtype is float32 or float16, ... (type of element)
        _dtype = self.TORCH_DTYPE_TO_ID[_tensor.dtype]
        _dtype = torch.tensor(_dtype, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _dtype, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _requires_grad = torch.tensor(
            1 if _tensor.requires_grad else 0,
            dtype=torch.long,
            device=self.device,
        )
        assert self.mpu.p2p(
            _requires_grad, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _ndims = len(_tensor.size())
        _ndims = torch.tensor(_ndims, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _ndims, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _shape = list(_tensor.size())
        _shape = torch.tensor(_shape, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _shape, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        if _tensor.dtype == torch.bool:
            _tensor = _tensor.half()

        if not _tensor.is_contiguous():
            _tensor = _tensor.contiguous()

        assert self.mpu.p2p(
            _tensor, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

    def recv_tensor(self, send_stage):
        """Receive tensor from other stage"""
        _dtype = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _dtype, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _dtype = self.TORCH_ID_TO_DTYPE[_dtype.item()]

        _requires_grad = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _requires_grad, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _requires_grad = True if _requires_grad.item() == 1 else False

        _ndims = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _ndims, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _shape = torch.tensor([0] * _ndims.item(), dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _shape, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _shape = tuple(_shape.tolist())

        if _dtype == torch.bool:
            __dtype = torch.half
        else:
            __dtype = _dtype

        recv_tensor = torch.zeros(size=_shape, dtype=__dtype, device=self.device)
        recv_tensor.requires_grad = _requires_grad and recv_tensor.is_floating_point()

        assert self.mpu.p2p(
            recv_tensor, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"

        if _dtype == torch.bool:
            recv_tensor = recv_tensor.bool()

        return recv_tensor

    def send_list(self, _list, recv_stage, send_type=False):
        """Send list to other stage"""
        assert isinstance(_list, list), f"wrong type: {_list} must be {list} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](list, recv_stage)

        list_len = len(_list)
        self.send_int(list_len, recv_stage)

        for item in _list:
            _type = type(item)
            assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"
            self.INSTRUCTIONS[_type]["send"](item, recv_stage, send_type=True)

    def recv_list(self, send_stage):
        """Receive list from other stage"""
        len_list = self.recv_int(send_stage)
        output_list = []

        for _ in range(len_list):
            _type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _recv = self.INSTRUCTIONS[_type]["recv"](send_stage)
            output_list.append(_recv)

        return output_list

    def send_set(self, _set, recv_stage, send_type=False):
        """Send set to other stage"""
        assert isinstance(_set, set), f"wrong type: {_set} must be {set} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](set, recv_stage)

        self.send_list(list(_set), recv_stage, False)

    def recv_set(self, send_stage):
        """Receive set from other stage"""
        return set(self.recv_list(send_stage))

    def send_tuple(self, _tuple, recv_stage, send_type=False):
        """Send tuple to other stage"""
        assert isinstance(_tuple, tuple), f"wrong type: {_tuple} must be {tuple} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](tuple, recv_stage)

        self.send_list(list(_tuple), recv_stage, send_type=False)

    def recv_tuple(self, send_stage):
        """Receive tuple from other stage"""
        return tuple(self.recv_list(send_stage))

    def send_size(self, _size, recv_stage, send_type=False):
        """Send torch.Size to other stage"""
        assert isinstance(_size, Size), f"wrong type: {_size} must be {Size} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](Size, recv_stage)

        self.send_list(list(_size), recv_stage, send_type=False)

    def recv_size(self, send_stage):
        """Receive torch.Size from other stage"""
        return Size(self.recv_list(send_stage))

    def send_dict(self, _dict, recv_stage, send_type=False):
        """Send dict to other stage"""
        assert isinstance(_dict, dict), f"wrong type: {_dict} must be {dict} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](dict, recv_stage)

        dict_len = len(_dict)
        self.send_int(dict_len, recv_stage)

        for key, val in _dict.items():
            _type_key, _type_val = type(key), type(val)
            assert _type_key in self.ID_TO_DTYPE, f"unsupported type: {_type_key}"
            assert _type_val in self.ID_TO_DTYPE, f"unsupported type: {_type_val}"
            self.INSTRUCTIONS[_type_key]["send"](key, recv_stage, send_type=True)
            self.INSTRUCTIONS[_type_val]["send"](val, recv_stage, send_type=True)

    def recv_dict(self, send_stage):
        """Receive dict from other stage"""
        len_dict = self.recv_int(send_stage)
        output_dict = {}

        for _ in range(len_dict):
            _key_type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _key_recv = self.INSTRUCTIONS[_key_type]["recv"](send_stage)
            _val_type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _val_recv = self.INSTRUCTIONS[_val_type]["recv"](send_stage)
            output_dict[_key_recv] = _val_recv

        return output_dict
