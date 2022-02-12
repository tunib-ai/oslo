import operator
from copy import deepcopy
from functools import reduce, partial

import torch
from tqdm import tqdm
import torch.distributed as dist

from oslo.pytorch.kernel_fusion.compile.aot_autograd import (
    _named_parameters,
    _named_buffers,
)
from oslo.pytorch.kernel_fusion.manage import GLOBAL_GRAPH_STORAGE, COMPILED_GRAPHS
from oslo.pytorch.kernel_fusion.manage.aot import AOTManager
from oslo.pytorch.kernel_fusion.params import ModuleManager
from oslo.pytorch.utils.kernel_fusion_mapping import KernelFusionMapping
from logging import getLogger

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

        for _, translated_params in module_translated_params.items():
            modules = self.get_modules(
                model=self.model,
                module_cls=translated_params.module_cls,
                module_name=translated_params.module_name,
                model_cls=translated_params.model_cls,
            )

            for module in modules:
                if (
                    translated_params.module_name,
                    translated_params.module_cls,
                    translated_params.model_cls,
                ) in module2graph:

                    graph = module2graph[
                        (
                            translated_params.module_name,
                            translated_params.module_cls,
                            translated_params.model_cls,
                        )
                    ]

                    module.forward = partial(graph, self=module)

    def request_compile(
        self,
        module,
        translated_params,
        batch_size,
        memory_efficient_fusion,
    ):
        desc = f"Creating AOT graphs for Fused{module.__class__.__qualname__}, bsz={batch_size}"
        max_length = translated_params.method.max_len

        for new_seq_len in tqdm(range(1, max_length + 1), desc=desc):
            new_translated_params_cls = KernelFusionMapping().get_mapping(self.model)
            new_translated_params_obj = new_translated_params_cls(
                self.model, batch_size, new_seq_len
            )
            new_translated_params = new_translated_params_obj.translated_params()

            for _new_translated_params in new_translated_params.values():
                if _new_translated_params == translated_params:
                    aot = AOTManager(self.model, batch_size, new_seq_len)
                    new_graph_info = self.graph_info(
                        module_cls=_new_translated_params.module_cls,
                        module_name=_new_translated_params.module_name,
                        model_cls=_new_translated_params.model_cls,
                        batch_size=batch_size,
                        seq_len=new_seq_len,
                    )

                    for params in _new_translated_params.params:
                        aot.compile(
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

            signature = module2signature[
                (
                    translated_params.module_name,
                    translated_params.module_cls,
                    translated_params.model_cls,
                )
            ]

            graph_candidates = self._get_graph_candidates(graph_info)
            module2graph[
                (
                    translated_params.module_name,
                    translated_params.module_cls,
                    translated_params.model_cls,
                )
            ] = self._create_optimized_forward(
                graph_candidates,
                signature,
            )

        return module2graph

    def logging_rank_0(self, msg):
        if dist.is_initialized() and dist.is_available():
            if dist.get_rank() == 0:
                logger.warning(msg)
        else:
            logger.warning(msg)

    def _create_optimized_forward(self, graph_candidates, orig_forward_parameters):
        def optimized_forward(*args, **kwargs):
            param_dict = self.get_param_dict(
                *args, **kwargs, orig_forward_parameters=orig_forward_parameters
            )
            most_similar_graph = self._get_most_similar_graph(
                param_dict, graph_candidates
            )

            if most_similar_graph["graph"] not in COMPILED_GRAPHS:
                COMPILED_GRAPHS.append(most_similar_graph["graph"])
                self.logging_rank_0(
                    f"JIT compiling new graph of Fused{most_similar_graph['module_cls']} "
                    f"for the {most_similar_graph['params']}."
                )

            module = kwargs.pop("self", None)
            return most_similar_graph["graph"](
                dict(_named_parameters(module, remove_duplicate=False)),
                dict(_named_buffers(module, remove_duplicate=False)),
                **{
                    key: param[0]
                    for key, param in param_dict.items()
                    if param[1] is False
                },
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

    @staticmethod
    def _get_most_similar_graph(param_dict, graph_candidates):
        most_similar_graphs = []
        for candidate in graph_candidates:
            if list(candidate["params"].keys()) != [
                k for k, v in param_dict.items() if v[1] is False
            ]:
                continue

            diff = 0
            for input_name, (input_value, is_default) in param_dict.items():
                for name, param in candidate["params"].items():
                    if name == input_name and isinstance(input_value, torch.Tensor):
                        diff -= (
                            reduce(operator.mul, param.size, 1)
                            - input_value.size().numel()
                        )

            most_similar_graphs.append(
                {
                    "module_cls": candidate["module_cls"],
                    "graph": candidate["graph"],
                    "params": candidate["params"],
                    "diff": diff,
                }
            )

        assert len(most_similar_graphs) != 0, "There are no graph candidates!"
        min_diff = min([graph["diff"] for graph in most_similar_graphs])
        most_similar_graph = None

        for graph in most_similar_graphs:
            if graph["diff"] == min_diff:
                most_similar_graph = graph

        return most_similar_graph
