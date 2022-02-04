import mmap

import torch
import torch.distributed as dist

from oslo.pytorch.activation_checkpointing.utils.utils import (
    is_activation_to_checkpoint,
)


class CheckpointPartitioner(object):
    def __init__(
        self,
        mpu,
        num_layers,
        cpu_checkpointing,
        partitioned_checkpointing,
        contiguous_checkpointing,
    ):
        self.mp_size = mpu.get_tensor_parallel_world_size() if mpu is not None else 1
        self.mp_rank = mpu.get_tensor_parallel_rank() if mpu is not None else 0
        self.mp_group = mpu.get_tensor_parallel_group() if mpu is not None else None

        self.num_layers = num_layers
        self.cpu_checkpointing = cpu_checkpointing
        self.partitioned_checkpointing = partitioned_checkpointing
        self.contiguous_checkpointing = contiguous_checkpointing

        self.contiguous_data_buffers = []
        self.contiguous_size_buffers = []
        self.data_offsets = []
        self.size_offsets = []

    def get_partition_start(self, item):
        size = item.numel()
        partition_size = size / self.mp_size
        start = partition_size * self.mp_rank
        return int(start)

    def get_partition_size(self, item):
        size = item.numel()
        assert size % self.mp_size == 0, (
            "Doesn't handle if partition activation "
            "if item is not divisible by mp size."
        )
        partition_size = size / self.mp_size
        return int(partition_size)

    def make_partitioned_activations(self, args):
        inputs = []
        num_non_ckpt_tensors = 0

        for arg_index, item in enumerate(args):
            if not is_activation_to_checkpoint(item, self.mp_size):
                inputs.append(item)
                num_non_ckpt_tensors += 1
                continue

            i = arg_index - num_non_ckpt_tensors
            partition_size = self.get_partition_size(item)

            partition = (
                item.detach()
                .contiguous()
                .view(-1)
                .narrow(0, self.get_partition_start(item), partition_size)
                .clone()
            )

            buffer_device = (
                torch.device("cpu") if self.cpu_checkpointing else partition.device
            )

            if self.contiguous_checkpointing:
                if i >= len(self.contiguous_data_buffers):
                    tensor_list = [
                        torch.tensor(()).new_empty(
                            [partition_size],
                            dtype=partition.dtype,
                            device=buffer_device,
                        )
                        for _ in range(self.num_layers)
                    ]
                    self.contiguous_data_buffers.append(tensor_list)
                    self.data_offsets.append(0)

                elif self.contiguous_data_buffers[i] is None:
                    tensor_list = [
                        torch.tensor(()).new_empty(
                            [partition_size],
                            dtype=partition.dtype,
                            device=buffer_device,
                        )
                        for _ in range(self.num_layers)
                    ]
                    self.contiguous_data_buffers[i] = tensor_list
                    self.data_offsets[i] = 0

                # Because the 'new_empty' returns uninitialized pages,
                # the pages need to be populated during the cudaMemcpy time
                # which increases the data copy time. To avoid this, we
                # pre-populate these pages by simply writing 0 ahead of
                # the actual cudaMemcpy operation time. Due to the
                # previously launched GPU kernels, there is a small
                # window of time here for CPUs to populate pages asynchronously.
                self.contiguous_data_buffers[i][self.data_offsets[i]].data[
                    range(
                        0,
                        self.contiguous_data_buffers[i][
                            self.data_offsets[i]
                        ].data.shape[0],
                        int(
                            mmap.PAGESIZE
                            / self.contiguous_data_buffers[i][
                                self.data_offsets[i]
                            ].data.element_size()
                        ),
                    )
                ] = 0

                contiguous_partition = self.contiguous_data_buffers[i][
                    self.data_offsets[i]
                ].data.copy_(partition.data)
                self.data_offsets[i] = self.data_offsets[i] + 1
                inputs.append(contiguous_partition)
            else:
                partition = partition.cpu() if self.cpu_checkpointing else partition
                inputs.append(partition)

        return inputs

    def gather_partitioned_activations(self, tensors, device=None):
        assert (
            len(tensors) % 2 == 0
        ), f"Expected even count of tensors, instead got {len(tensors)}"

        inputs = []
        num_args = int(len(tensors) / 2)

        for i in range(num_args):
            item = tensors[2 * i]
            size = tensors[2 * i + 1]

            if not is_activation_to_checkpoint(item, self.mp_size):
                inputs.append(item)
                continue

            partition_size = item.numel()
            tensor_size = partition_size * self.mp_size

            if device is not None:
                flat_tensor = torch.zeros(
                    [tensor_size],
                    dtype=item.dtype,
                    device=device,
                )
            else:
                flat_tensor = torch.zeros(
                    [tensor_size],
                    dtype=item.dtype,
                    device=item.device,
                )

            partitions = []
            for i in range(self.mp_size):
                part_i = flat_tensor.narrow(0, partition_size * i, partition_size)
                if i == self.mp_rank:
                    part_i.copy_(item)
                partitions.append(part_i)

            if self.mp_group is not None:
                dist.all_gather(
                    partitions,
                    partitions[self.mp_rank],
                    group=self.mp_group,
                )

            input_tensor = flat_tensor.view(list(size.numpy()))
            item.data = input_tensor.data
            inputs.append(item)

        return tuple(inputs)

    @staticmethod
    def merge_tensors(tensor_objects, non_tensor_objects, tensor_flags):
        tensor_idx = 0
        non_tensor_idx = 0
        merged_objects = []

        for is_tensor in tensor_flags:
            if is_tensor:
                merged_objects.append(tensor_objects[tensor_idx])
                tensor_idx += 1
            else:
                merged_objects.append(non_tensor_objects[non_tensor_idx])
                non_tensor_idx += 1

        return tuple(merged_objects)

    def get_partitioned_activations_for_backward(self, args, inputs):
        new_args = []
        num_non_ckpt_tensors = 0

        for arg_index, (arg, inp) in enumerate(zip(args, inputs)):
            size = torch.tensor(arg.size()) if torch.is_tensor(arg) else None

            if not is_activation_to_checkpoint(arg, self.mp_size):
                new_args.append(arg)
                new_args.append(size)
                num_non_ckpt_tensors += 1
                continue

            arg.data = inp.data
            new_args.append(arg)
            i = arg_index - num_non_ckpt_tensors

            if self.contiguous_checkpointing:
                numel = size.numel()
                if i >= len(self.contiguous_size_buffers):
                    tmp = torch.tensor(())
                    self.contiguous_size_buffers.append(
                        tmp.new_empty(
                            [numel * self.num_layers],
                            dtype=size.dtype,
                            device=size.device,
                        )
                    )
                    self.size_offsets.append(0)
                elif self.contiguous_size_buffers[i] is None:
                    tmp = torch.tensor(())
                    self.contiguous_size_buffers[i] = tmp.new_empty(
                        [numel * self.num_layers],
                        dtype=size.dtype,
                        device=size.device,
                    )
                    self.size_offsets[i] = 0

                contiguous_size = (
                    self.contiguous_size_buffers[i]
                    .narrow(0, self.size_offsets[i], numel)
                    .data.copy_(size.data)
                )
                contiguous_size = contiguous_size.view_as(size)
                self.size_offsets[i] = self.size_offsets[i] + numel
                new_args.append(contiguous_size)
            else:
                new_args.append(size)

        return new_args

    def get_cpu_activations_for_backward(self, args, inputs):
        new_args = []
        for i, (arg, inp) in enumerate(zip(args, inputs)):
            if not is_activation_to_checkpoint(arg, self.mp_size):
                new_args.append(arg)
                continue

            arg.data = inp.data
            new_args.append(arg)

        return new_args
