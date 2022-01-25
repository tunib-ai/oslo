import copy
from typing import List

import torch
import torch.distributed as dist
from torch.fx import Node
from torch.nn import Linear, Module, ModuleList, Embedding
from transformers.modeling_utils import Conv1D
from transformers.utils.fx import symbolic_trace

from oslo.modeling_utils import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from oslo.parallelism.utils import rgetattr, rsetattr


class TensorParallelEngine(object):
    def __init__(self, model, mpu, info):
        self.model = model
        self.mpu = mpu
        self.info = info
        self.module_lists = []
        self.device = torch.cuda.current_device()

        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                self.module_lists.append((name, module))

        assert (
            self.module_lists is not None
        ), "There are no parallelizable modules. please check the model."

    def _reduce_arguments(self):
        for name, module_list in self.module_lists:
            for module in module_list.modules():
                for arg in self.info.reducing_required():
                    if hasattr(module, arg):
                        reduced_arg = (
                            getattr(module, arg)
                            // self.mpu.get_tensor_parallel_world_size()
                        )
                        setattr(module, arg, reduced_arg)

    @staticmethod
    def _is_module(node: Node):
        return (
            node.op not in ["placeholder", "call_function", "call_method"]
            and "tensor_constant" not in node.target
        )

    @staticmethod
    def _get_fusion_degree(module):
        if isinstance(module, Linear) or isinstance(module, Conv1D):
            return max(module.weight.size()[0], module.weight.size()[1]) // min(
                module.weight.size()[0], module.weight.size()[1]
            )
        else:
            return 1

    def _get_parallelizable_module(self, module, compute_fusion_degree):
        if isinstance(module, ColumnParallelLinear) or isinstance(
            module, RowParallelLinear
        ):
            # already parallelized
            return None

        else:
            fusion_degree = (
                self._get_fusion_degree(module) if compute_fusion_degree else 1
            )

            if isinstance(module, Linear):
                return {
                    "module": module,
                    "reversed": False,
                    "fusion_degree": fusion_degree,
                }
            elif isinstance(module, Conv1D):
                return {
                    "module": module,
                    "reversed": True,
                    "fusion_degree": fusion_degree,
                }
            else:
                return None

    def _slice(
        self,
        module,
        reversed,
        fusion_degree,
        slice_bias,
        parallel_dim,
        to_gpu,
    ):
        parallel_dim = parallel_dim if not reversed else abs(parallel_dim - 1)
        world_size = self.mpu.get_tensor_parallel_world_size()
        gpu_index = self.mpu.get_tensor_parallel_rank()
        update_attributes = {
            "tied_with_embedding": False,
            "reversed": reversed,
            "fusion_degree": fusion_degree,
            "orig_module": copy.deepcopy(module.__class__),
        }

        if hasattr(module, "weight"):
            if module.weight.dim() >= 1:
                weight = module.weight.chunk(
                    chunks=fusion_degree * world_size,
                    dim=parallel_dim,
                )
                if fusion_degree > 1:
                    weight = self._realign_fused_tensors(weight, world_size)
                module.weight.data = weight[gpu_index].contiguous()

            if to_gpu is True:
                module.weight.data = module.weight.to(self.device)

            update_attributes["in_features"] = module.weight.size()[0]
            update_attributes["out_features"] = module.weight.size()[1]

        if hasattr(module, "bias"):
            if slice_bias is True and module.bias.dim() >= 1:
                bias = module.bias.chunk(
                    chunks=fusion_degree * world_size,
                    dim=0,
                )
                if fusion_degree > 1:
                    bias = self._realign_fused_tensors(bias, world_size)
                module.bias.data = bias[gpu_index].contiguous()

            if to_gpu is True:
                module.bias.data = module.bias.to(self.device)

        return module, update_attributes

    @staticmethod
    def _realign_fused_tensors(tensor, world_size):
        ranks = (len(tensor) + world_size - 1) // world_size
        tensor = [tensor[i * world_size : (i + 1) * world_size] for i in range(ranks)]
        tensor = list(map(lambda x: torch.cat([*x], dim=-1), zip(*tensor)))
        return tensor

    def _column_slice(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
        to_gpu: bool,
    ) -> Module:
        sliced_module, attributes = self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=True,
            parallel_dim=0,
            to_gpu=to_gpu,
        )

        for k, v in attributes.items():
            setattr(sliced_module, k, v)

        sliced_module.__class__ = ColumnParallelLinear

        return sliced_module

    def _row_slice(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
        to_gpu: bool,
    ) -> List[Module]:
        sliced_module, attributes = self._slice(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            slice_bias=False,
            parallel_dim=1,
            to_gpu=to_gpu,
        )

        for k, v in attributes.items():
            setattr(sliced_module, k, v)

        sliced_module.__class__ = RowParallelLinear

        return sliced_module

    @torch.no_grad()
    def _realign_fused_tensors(self, tensors, fusion_degree, dim):
        result_tensors = {i: [] for i in range(fusion_degree)}

        for tensor in tensors:
            chunks = tensor.chunk(fusion_degree, dim=dim)
            for i, chunk in enumerate(chunks):
                result_tensors[i].append(chunk)

        for key, val in result_tensors.items():
            result_tensors[key] = torch.cat(val, dim=dim)

        return torch.cat(list(result_tensors.values()), dim=dim)

    def _parallelize_module(self, module_name, traced, compute_fusion_degree):
        parallelizable_modules = []
        parallelizable_targets = []

        for node in traced.graph.nodes:
            if self._is_module(node) and module_name in node.target and node.target:
                parallelizable_target = node.target
                parallelizable_target = parallelizable_target.replace(".weight", "")
                parallelizable_target = parallelizable_target.replace(".bias", "")

                if parallelizable_target not in parallelizable_targets:
                    parallelizable_targets.append(parallelizable_target)

        for target in parallelizable_targets:
            _module = rgetattr(self.model, target, None)
            if _module is not None:
                parallelizable_module = self._get_parallelizable_module(
                    _module,
                    compute_fusion_degree=compute_fusion_degree,
                )
                if parallelizable_module is not None:
                    parallelizable_modules.append(parallelizable_module)

        column_parallel_modules = parallelizable_modules[:-1]
        row_parallel_module = parallelizable_modules[-1]

        for column_parallel_module in column_parallel_modules:
            self._column_slice(
                module=column_parallel_module["module"],
                reversed=column_parallel_module["reversed"],
                fusion_degree=column_parallel_module["fusion_degree"],
                to_gpu=True,
                # TODO: We should consider about pipeline parallelism
            )

        self._row_slice(
            module=row_parallel_module["module"],
            reversed=row_parallel_module["reversed"],
            fusion_degree=row_parallel_module["fusion_degree"],
            to_gpu=True,
            # TODO: We should consider about pipeline parallelism
        )

    def _parallelize_module_list(self, module_list, name, traced):
        for i, modules in enumerate(module_list):
            if hasattr(self.model.config, "num_attention_heads"):
                assert (
                    self.model.config.num_attention_heads
                    >= self.mpu.get_tensor_parallel_world_size()
                ), "number of attention heads must be bigger than tensor parallel size."
            if hasattr(self.model.config, "hidden_size"):
                assert (
                    self.model.config.hidden_size
                    >= self.mpu.get_tensor_parallel_world_size()
                ), "hidden size must be bigger than tensor parallel size."

            for _name, _module in modules.named_modules():
                module_name = f"{name}.{i}.{_name}"
                if isinstance(_module, self.info.attention()):
                    self._parallelize_module(
                        module_name=module_name,
                        traced=traced,
                        compute_fusion_degree=True,
                    )
                elif isinstance(_module, self.info.mlp()):
                    self._parallelize_module(
                        module_name=module_name,
                        traced=traced,
                        compute_fusion_degree=False,
                    )

            for _name, _module in modules.named_modules():
                parallelizable_module = self._get_parallelizable_module(
                    _module, compute_fusion_degree=False
                )

                if parallelizable_module is not None:
                    self._column_slice(
                        module=parallelizable_module["module"],
                        reversed=parallelizable_module["reversed"],
                        fusion_degree=parallelizable_module["fusion_degree"],
                        to_gpu=True,
                        # TODO: We should consider about pipeline parallelism
                    )

    def _parallelize_word_embedding(self):
        embedding = self.model.get_input_embeddings()
        chunked_weight = torch.chunk(
            embedding.weight,
            chunks=self.mpu.get_tensor_parallel_world_size(),
            dim=0,
        )
        embedding.weight.data = chunked_weight[self.mpu.get_tensor_parallel_rank()]
        embedding.weight.data = embedding.weight.to(self.device)
        embedding.num_embeddings = embedding.weight.size()[0]

        setattr(embedding, "orig_module", copy.deepcopy(embedding.__class__))
        embedding.__class__ = VocabParallelEmbedding

    def _parallelize_tied_head(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Linear) or isinstance(module, Conv1D):
                if module.weight is self.model.get_input_embeddings().weight:
                    setattr(module, "reversed", isinstance(module, Conv1D))
                    setattr(module, "fusion_degree", self._get_fusion_degree(module))
                    setattr(module, "orig_module", copy.deepcopy(module.__class__))
                    setattr(module, "tied_with_embedding", True)
                    module.__class__ = ColumnParallelLinear

    def parallelize(self):
        traced = symbolic_trace(self.model)
        for name, module_list in self.module_lists:
            self._parallelize_module_list(module_list, name, traced=traced)

        self._parallelize_word_embedding()
        self._parallelize_tied_head()
        self._reduce_arguments()

        for k, v in dict(self.model.state_dict()).items():
            if not v.is_cuda:
                if torch.is_tensor(v):
                    rsetattr(self.model, k + ".data", v.to(self.device))


class TensorDeparallelEngine(object):
    def __init__(self, model, mpu, info):
        self.model = model
        self.mpu = mpu
        self.info = info
        self.module_lists = []
        self.device = torch.cuda.current_device()

        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                self.module_lists.append((name, module))

        assert (
            self.module_lists is not None
        ), "There are no parallelizable modules. please check the model."

    @torch.no_grad()
    def _realign_fused_tensors(self, tensors, fusion_degree, dim):
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
        parallel_dim,
    ):
        parallel_dim = parallel_dim if not reversed else abs(parallel_dim - 1)
        world_size = self.mpu.get_tensor_parallel_world_size()
        update_attributes = {}

        if hasattr(module, "weight"):
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
                    output = self._realign_fused_tensors(
                        tensors=tensor_list,
                        fusion_degree=fusion_degree,
                        dim=parallel_dim,
                    )
                else:
                    output = torch.cat(
                        tensor_list,
                        dim=parallel_dim,
                    )

                module.weight.data = output
                update_attributes["in_features"] = module.weight.size()[0]
                update_attributes["out_features"] = module.weight.size()[1]
                update_attributes["nx"] = module.weight.size()[0]
                update_attributes["nf"] = module.weight.size()[1]

            module.weight.data = module.weight.cpu()

        if hasattr(module, "bias"):
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
                    output = self._realign_fused_tensors(
                        tensors=tensor_list,
                        fusion_degree=fusion_degree,
                        dim=0,
                    )
                else:
                    output = torch.cat(
                        tensor_list,
                        dim=0,
                    )

                module.bias.data = output

            module.bias.data = module.bias.cpu()

        return module, update_attributes

    def _column_merge(
        self,
        module: Module,
        fusion_degree: int,
        reversed: bool,
    ) -> Module:
        merged_module, attributes = self._merge(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            merge_bias=True,
            parallel_dim=0,
        )

        for k, v in attributes.items():
            setattr(merged_module, k, v)

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
        merged_module, attributes = self._merge(
            module=module,
            reversed=reversed,
            fusion_degree=fusion_degree,
            merge_bias=False,
            parallel_dim=1,
        )

        for k, v in attributes.items():
            setattr(merged_module, k, v)

        if hasattr(module, "orig_module"):
            module.__class__ = module.orig_module
        else:
            module.__class__ = Conv1D if reversed else Linear

        return merged_module

    def _deparallelize_word_embedding(self):
        embedding = self.model.get_input_embeddings()
        if not embedding.weight.is_cuda:
            embedding.weight.data = embedding.weight.cuda()

        gathered_weight = self.mpu._gather(embedding.weight, dim=0)
        embedding.weight.data = gathered_weight.cpu()
        embedding.num_embeddings = embedding.weight.size()[0]

        if hasattr(embedding, "orig_module"):
            embedding.__class__ = embedding.orig_module
        else:
            embedding.__class__ = Embedding

    def _deparallelize_tied_head(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ColumnParallelLinear):
                if module.weight is self.model.get_input_embeddings().weight:
                    if hasattr(module, "orig_module"):
                        module.__class__ = module.orig_module
                    else:
                        if hasattr(module, "reversed"):
                            module.__class__ = Conv1D if module.reversed else Linear
                        else:
                            module.__class__ = Linear

                    module.weight.data = module.weight.cpu()

    def _deparallelize_module_list(self, module_list):
        for modules in module_list:
            for module in modules.modules():
                if isinstance(module, ColumnParallelLinear):
                    self._column_merge(
                        module=module,
                        fusion_degree=module.fusion_degree,
                        reversed=module.reversed,
                    )
                elif isinstance(module, RowParallelLinear):
                    self._row_merge(
                        module=module,
                        fusion_degree=module.fusion_degree,
                        reversed=module.reversed,
                    )

    def _restore_arguments(self):
        for name, module_list in self.module_lists:
            for module in module_list.modules():
                for arg in self.info.reducing_required():
                    if hasattr(module, arg):
                        reduced_arg = (
                            getattr(module, arg)
                            * self.mpu.get_tensor_parallel_world_size()
                        )
                        setattr(module, arg, reduced_arg)

    def deparallelize(self):
        for _, module_list in self.module_lists:
            self._deparallelize_module_list(module_list)

        self._deparallelize_word_embedding()
        self._deparallelize_tied_head()
        self._restore_arguments()

        for k, v in dict(self.model.state_dict()).items():
            if v.is_cuda:
                if torch.is_tensor(v):
                    rsetattr(self.model, k + ".data", v.cpu())
