import operator
from functools import reduce, partial
from logging import getLogger

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm

from oslo.pytorch.kernel_fusion.compile.aot_autograd import (
    _named_parameters,
    _named_buffers,
)
from oslo.pytorch.kernel_fusion.compile.compat._stateless import reparametrize_module
from oslo.pytorch.kernel_fusion.graphs import GLOBAL_GRAPH_STORAGE, COMPILED_GRAPHS
from oslo.pytorch.kernel_fusion.graphs.generator import GraphGenerator
from oslo.pytorch.kernel_fusion.params import ModuleManager
from oslo.pytorch.utils.kernel_fusion_mapping import KernelFusionMapping

logger = getLogger(__name__)


class GraphReplacer(ModuleManager):
    def __init__(self, model, fuser):
        self.model = model
        self.fuser = fuser

    def graph_info(self, module_cls, module_name, model_cls, batch_size, seq_len):
        return {
            "module_cls": module_cls,
            "module_name": module_name,
            "model_cls": model_cls,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "fuser": self.fuser,
            "device": str(self.model.device),
        }

    def replace_graph(
        self,
        batch_size,
        seq_len,
        module_translated_params,
        module2signature,
        memory_efficient_fusion,
    ):
        module2graph = self.make_module2graph(
            batch_size,
            seq_len,
            module_translated_params,
            module2signature,
            memory_efficient_fusion,
        )

        for module, translated_params in module_translated_params.items():
            graph_key = translated_params.key()
            if graph_key in module2graph:
                graph = module2graph[graph_key]
                module.forward = graph

    def request_compile(
        self,
        module,
        translated_params,
        batch_size,
        memory_efficient_fusion,
    ):
        desc = f"Generating graphs of Fused{module.__class__.__qualname__} for batch_size={batch_size}"
        max_length = translated_params.method.max_len

        for new_seq_len in tqdm(range(1, max_length + 1), desc=desc):
            new_translated_params_cls = KernelFusionMapping().get_mapping(self.model)
            new_translated_params_obj = new_translated_params_cls(
                self.model, batch_size, new_seq_len
            )
            new_translated_params = new_translated_params_obj.translated_params()

            for _new_translated_params in new_translated_params.values():
                if _new_translated_params == translated_params:
                    generator = GraphGenerator(self.model, batch_size, new_seq_len)
                    new_graph_info = self.graph_info(
                        module_cls=_new_translated_params.module_cls,
                        module_name=_new_translated_params.module_name,
                        model_cls=_new_translated_params.model_cls,
                        batch_size=batch_size,
                        seq_len=new_seq_len,
                    )

                    for params in _new_translated_params.params:
                        generator.generate(
                            module=module,
                            param=params,
                            graph_info=new_graph_info,
                            memory_efficient_fusion=memory_efficient_fusion,
                        )

    def make_module2graph(
        self,
        batch_size,
        seq_len,
        module_translated_params,
        module2signature,
        memory_efficient_fusion,
    ):
        module2graph = {}
        for module, translated_params in module_translated_params.items():
            graph_info = self.graph_info(
                module_cls=translated_params.module_cls,
                module_name=translated_params.module_name,
                model_cls=translated_params.model_cls,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            graph_candidates = self._get_graph_candidates(graph_info)

            if len(graph_candidates) == 0:
                self.request_compile(
                    module,
                    translated_params,
                    batch_size,
                    memory_efficient_fusion,
                )

            graph_key = translated_params.key()
            signature = module2signature[graph_key]
            graph_candidates = self._get_graph_candidates(graph_info)

            module2graph[graph_key] = self._create_optimized_forward(
                graph_candidates, signature
            )

        return module2graph

    @staticmethod
    def logging_rank_0(msg):
        if dist.is_initialized() and dist.is_available():
            if dist.get_rank() == 0:
                logger.warning(msg)
        else:
            logger.warning(msg)

    def _create_optimized_forward(self, graph_candidates, orig_forward_parameters):
        def optimized_forward(*args, **kwargs):
            named_parameters = kwargs.pop("named_parameters", None)
            named_buffers = kwargs.pop("named_buffers", None)

            param_dict = self.get_param_dict(
                *args, **kwargs, orig_forward_parameters=orig_forward_parameters
            )

            most_similar_graph = self._get_most_similar_graph(
                param_dict, graph_candidates
            )

            if most_similar_graph["graph"] not in COMPILED_GRAPHS:
                COMPILED_GRAPHS.append(most_similar_graph["graph"])
                self.logging_rank_0(
                    f"Compiling graph of Fused{most_similar_graph['module_cls']} "
                    f"for the {most_similar_graph['params']}."
                )

            non_default_input = {
                key: param[0] for key, param in param_dict.items() if param[1] is False
            }

            return most_similar_graph["graph"](
                **non_default_input,
                named_parameters=named_parameters,
                named_buffers=named_buffers,
            )

        return optimized_forward

    @staticmethod
    def _get_graph_candidates(graph_info):
        candidates = []
        for graph in GLOBAL_GRAPH_STORAGE:
            graph_matches = [False] * len(graph_info)
            for i, (k, v) in enumerate(graph_info.items()):
                if graph[k] == v:
                    graph_matches[i] = True
            if all(graph_matches):
                candidates.append(graph)

        return candidates
