# Copyright 2021 TUNiB Inc.

import os
from abc import ABC
from dataclasses import dataclass
from math import floor
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


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

    # World size

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


@dataclass
class Layer:
    """Data class to describe a layer in the model"""

    module: nn.Module = None
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    replace: dict = None
    n_fused: int = None
    reversed: bool = None
    parallel: bool = True
    input_is_parallel: bool = True
    gather_output: bool = False
    tied_embedding: nn.Module = None


class LayerPolicy(ABC):
    """
    Layer policy for model parallelism and kernel fusion
    You can check more details here: https://github.com/tunib-ai/parallelformers/blob/main/POLICY.md

    References:
        The design of the LayerPolicy class is inspired by Microsoft DeepSpeed.
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py
    """

    @staticmethod
    def reduce_arguments(layer, world_size, config):
        ...

    @staticmethod
    def fused_modules():
        return {}

    @staticmethod
    def attn_qkv(layer, config):
        return []

    @staticmethod
    def attn_out(layer, config):
        return []

    @staticmethod
    def attn_norm(layer, config):
        return []

    @staticmethod
    def mlp_in(layer, config):
        return []

    @staticmethod
    def mlp_out(layer, config):
        return []

    @staticmethod
    def mlp_norm(layer, config):
        return []

    @staticmethod
    def word_embedding(model, config):
        return []

    @staticmethod
    def preblock_layers(model, config):
        return []

    @staticmethod
    def block_layers(model, config):
        return []

    @staticmethod
    def postblock_layers(model, config):
        return []

    @staticmethod
    def copy_to_all(layer, config):
        return []

    @staticmethod
    def original_layer_class():
        raise NotImplementedError
