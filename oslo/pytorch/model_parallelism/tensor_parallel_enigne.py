import copy
from typing import List

import torch
import torch.distributed as dist
from torch.nn import Embedding, Linear, Module

from oslo.pytorch.model_parallelism.utils.distributed import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from oslo.pytorch.model_parallelism.utils.mappings import (
    TensorParallelismMapping,
    update_module_arguments,
)


class TensorParallelEngine(object):
    def __init__(self, model, mpu, mapping=None):
        self.model = model
        self.mpu = mpu
        self.mapping = mapping if mapping is not None else TensorParallelismMapping()
        self.device = torch.cuda.current_device()

    def _update_mp_arguments(self):
        for module in self.model.modules():
            for elem in self.mapping.update_attrs(self.model):
                if hasattr(module, elem.name):
                    world_size = self.mpu.get_tensor_parallel_world_size()
                    reduced_arg = getattr(module, elem.name) // world_size
                    setattr(module, elem.name, reduced_arg)

    @staticmethod
    def _deconstruct_combined_qkv(tensor, world_size):
        ranks = (len(tensor) + world_size - 1) // world_size
        tensor = [tensor[i * world_size : (i + 1) * world_size] for i in range(ranks)]
        tensor = list(map(lambda x: torch.cat([*x], dim=-1), zip(*tensor)))
        return tensor

    def _slice(
        self,
        module,
        reversed,
        fusion_degree,
        slice_bias,
        dim,
        to_gpu,
    ):
        dim = dim if not reversed else abs(dim - 1)
        world_size = self.mpu.get_tensor_parallel_world_size()
        gpu_index = self.mpu.get_tensor_parallel_rank()

        update_module_arguments(
            module=module,
            mpu=self.mpu,
            reversed=reversed,
            fusion_degree=fusion_degree,
            orig_module=copy.deepcopy(module.__class__),
            gather_output=False,
        )

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                weight = module.weight.chunk(fusion_degree * world_size, dim=dim)
                if fusion_degree > 1:
                    weight = self._deconstruct_combined_qkv(weight, world_size)
                module.weight.data = weight[gpu_index].contiguous()

            if to_gpu is True:
                module.weight.data = module.weight.to(self.device)

            update_module_arguments(
                module=module,
                in_features=module.weight.size()[0],
                out_features=module.weight.size()[1],
            )

        if hasattr(module, "bias") and module.bias is not None:
            if slice_bias is True and module.bias.dim() >= 1:

                bias = module.bias.chunk(fusion_degree * world_size, dim=0)
                if fusion_degree > 1:
                    bias = self._deconstruct_combined_qkv(bias, world_size)
                module.bias.data = bias[gpu_index].contiguous()

            if to_gpu is True:
                module.bias.data = module.bias.to(self.device)

        return module

    def _column_slice(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
        to_gpu: bool,
    ) -> Module:
        return self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=True,
            to_gpu=to_gpu,
            dim=0,
        )

    def _row_slice(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
        to_gpu: bool,
    ) -> List[Module]:
        return self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=False,
            to_gpu=to_gpu,
            dim=1,
        )

    def _parallelize_embedding(self):
        from transformers import Conv1D

        embedding = self.model.get_input_embeddings()
        chunked_weights = torch.chunk(
            embedding.weight, chunks=self.mpu.get_tensor_parallel_world_size(), dim=0
        )
        chunked_weight = chunked_weights[self.mpu.get_tensor_parallel_rank()]
        embedding.weight.data = chunked_weight.to(self.device)
        update_module_arguments(
            module=embedding,
            mpu=self.mpu,
            orig_module=copy.deepcopy(embedding.__class__),
        )

        if isinstance(embedding, Embedding):
            embedding.__class__ = VocabParallelEmbedding

        for name, module in self.model.named_modules():
            if (
                hasattr(module, "weight")
                and module.weight is embedding.weight
                and not isinstance(module, Embedding)
            ):
                update_module_arguments(
                    module=module,
                    mpu=self.mpu,
                    reversed=self.mapping.is_reversed_param(self.model, name),
                    fusion_degree=1,
                    orig_module=copy.deepcopy(module.__class__),
                    gather_output=True,
                )

                if isinstance(module, Linear) or isinstance(module, Conv1D):
                    module.__class__ = ColumnParallelLinear

    def _parallelize_modules(self):
        from transformers import Conv1D

        for param_name, module in self.model.named_modules():
            if self.mapping.is_column_parallel(self.model, param_name):
                self._column_slice(
                    module=module,
                    reversed=self.mapping.is_reversed_param(self.model, param_name),
                    fusion_degree=self.mapping.get_combined_qkv_degree(
                        self.model, param_name, module
                    ),
                    to_gpu=True,
                )
                if isinstance(module, Linear) or isinstance(module, Conv1D):
                    module.__class__ = ColumnParallelLinear

            elif self.mapping.is_row_parallel(self.model, param_name):
                self._row_slice(
                    module=module,
                    reversed=self.mapping.is_reversed_param(self.model, param_name),
                    fusion_degree=1,
                    to_gpu=True,
                )
                if isinstance(module, Linear) or isinstance(module, Conv1D):
                    module.__class__ = RowParallelLinear

    def _postprocess(self):
        for param in self.model.parameters():
            if not param.is_cuda:
                if torch.is_tensor(param):
                    param.data = param.to(self.device)

        for param in self.model.buffers():
            if not param.is_cuda:
                if torch.is_tensor(param):
                    param.data = param.to(self.device)

    def parallelize(self):
        self._update_mp_arguments()
        self._parallelize_embedding()
        self._parallelize_modules()
        self._postprocess()
        update_module_arguments(self.model, mpu=self.mpu)


class TensorDeparallelEngine(object):
    def __init__(self, model, mpu, mapping=None):
        self.model = model
        self.mpu = mpu
        self.mapping = mapping if mapping is not None else TensorParallelismMapping()
        self.device = torch.cuda.current_device()

    def _update_mp_arguments(self):
        for module in self.model.modules():
            for elem in self.mapping.update_attrs(self.model):
                if hasattr(module, elem.name):
                    world_size = self.mpu.get_tensor_parallel_world_size()
                    reduced_arg = getattr(module, elem.name) * world_size
                    setattr(module, elem.name, reduced_arg)

    @torch.no_grad()
    def _reconstruct_combined_qkv(self, tensors, fusion_degree, dim):
        result_tensors = {i: [] for i in range(fusion_degree)}
        for tensor in tensors:
            chunks = tensor.chunk(fusion_degree, dim=dim)
            for i, chunk in enumerate(chunks):
                result_tensors[i].append(chunk)
        for key, val in result_tensors.items():
            result_tensors[key] = torch.cat(val, dim=dim)
        return torch.cat(list(result_tensors.values()), dim=dim)

    def _merge(
        self,
        module,
        reversed,
        fusion_degree,
        merge_bias,
        dim,
    ):
        dim = dim if not reversed else abs(dim - 1)
        world_size = self.mpu.get_tensor_parallel_world_size()
        update_module_arguments(module=module, mpu=None)

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.dim() >= 1:
                if not module.weight.is_contiguous():
                    module.weight.data = module.weight.contiguous()
                if not module.weight.is_cuda:
                    module.weight.data = module.weight.to(self.device)

                tensor_list = [
                    torch.zeros_like(module.weight) for _ in range(world_size)
                ]

                dist.all_gather(
                    tensor_list,
                    module.weight,
                    group=self.mpu.get_tensor_parallel_group(),
                )

                if fusion_degree > 1:
                    output = self._reconstruct_combined_qkv(
                        tensor_list, fusion_degree, dim
                    )
                else:
                    output = torch.cat(tensor_list, dim=dim)

                module.weight.data = output
                update_module_arguments(
                    module=module,
                    in_features=module.weight.size()[0],
                    out_features=module.weight.size()[1],
                    nx=module.weight.size()[0],
                    nf=module.weight.size()[1],
                )

            module.weight.data = module.weight.cpu()

        if hasattr(module, "bias") and module.bias is not None:
            if merge_bias is True and module.bias.dim() >= 1:
                if not module.bias.is_contiguous():
                    module.bias.data = module.bias.contiguous()
                if not module.bias.is_cuda:
                    module.bias.data = module.bias.to(self.device)

                tensor_list = [torch.zeros_like(module.bias) for _ in range(world_size)]

                dist.all_gather(
                    tensor_list,
                    module.bias,
                    group=self.mpu.get_tensor_parallel_group(),
                )

                if fusion_degree > 1:
                    output = self._reconstruct_combined_qkv(
                        tensor_list, fusion_degree, 0
                    )
                else:
                    output = torch.cat(tensor_list, dim=0)

                module.bias.data = output

            module.bias.data = module.bias.cpu()

        return module

    def _column_merge(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
    ) -> Module:
        from transformers import Conv1D

        merged_module = self._merge(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            merge_bias=True,
            dim=0,
        )

        if hasattr(module, "orig_module"):
            module.__class__ = module.orig_module
        else:
            module.__class__ = Conv1D if reversed else Linear

        return merged_module

    def _row_merge(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
    ) -> List[Module]:
        from transformers import Conv1D

        merged_module = self._merge(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            merge_bias=False,
            dim=1,
        )

        if hasattr(module, "orig_module"):
            module.__class__ = module.orig_module
        else:
            module.__class__ = Conv1D if reversed else Linear

        return merged_module

    def _deparallelize_embedding(self):
        from transformers import Conv1D

        embedding = self.model.get_input_embeddings()
        if not embedding.weight.is_cuda:
            embedding.weight.data = embedding.weight.cuda()

        gathered_weight = self.mpu._gather(embedding.weight, dim=0)
        embedding.weight.data = gathered_weight.cpu()

        update_module_arguments(
            module=embedding,
            mpu=None,
            num_embeddings=embedding.weight.size()[0],
        )

        if hasattr(embedding, "orig_module"):
            embedding.__class__ = embedding.orig_module
        else:
            embedding.__class__ = Embedding

        for module in self.model.modules():
            if (
                hasattr(module, "weight")
                and module.weight is embedding.weight
                and not isinstance(module, Embedding)
            ):
                update_module_arguments(
                    module=module,
                    mpu=None,
                    out_features=embedding.num_embeddings,
                )

                if hasattr(module, "orig_module"):
                    module.__class__ = module.orig_module
                else:
                    if hasattr(module, "reversed"):
                        module.__class__ = Conv1D if module.reversed else Linear
                    else:
                        module.__class__ = Linear

                module.weight.data = module.weight.cpu()

    def _deparallelize_modules(self):
        for param_name, module in self.model.named_modules():
            if self.mapping.is_column_parallel(self.model, param_name):
                self._column_merge(
                    module=module,
                    fusion_degree=module.fusion_degree,
                    reversed=module.reversed,
                )
            elif self.mapping.is_row_parallel(self.model, param_name):
                self._row_merge(
                    module=module,
                    fusion_degree=module.fusion_degree,
                    reversed=module.reversed,
                )

    def _postprocess(self):
        for param in self.model.parameters():
            if param.is_cuda:
                if torch.is_tensor(param):
                    param.data = param.cpu()

        for param in self.model.buffers():
            if param.is_cuda:
                if torch.is_tensor(param):
                    param.data = param.cpu()

    def deparallelize(self):
        self._update_mp_arguments()
        self._deparallelize_embedding()
        self._deparallelize_modules()
        self._postprocess()
        update_module_arguments(self.model, mpu=None)
