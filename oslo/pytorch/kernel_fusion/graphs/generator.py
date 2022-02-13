from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.pytorch.kernel_fusion.compile.aot_autograd import aot_function
from oslo.pytorch.kernel_fusion.compile.compat import _stateless
from oslo.pytorch.kernel_fusion.compile.compat._stateless import reparametrize_module
from oslo.pytorch.kernel_fusion.compile.compilers import (
    ts_compile,
    default_decompositions,
)
from oslo.pytorch.kernel_fusion.compile.partitioners import (
    min_cut_rematerialization_partition,
)
from oslo.pytorch.kernel_fusion.graphs import GLOBAL_GRAPH_STORAGE
from oslo.pytorch.kernel_fusion.params import TensorMeta


class GraphGenerator(object):
    def __init__(self, model, batch_size, seq_len):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len

    @staticmethod
    def compile_config(static_argnums, memory_efficient_fusion):
        config = {
            "fw_compiler": ts_compile,
            "bw_compiler": ts_compile,
            "hasher_type": "StaticShapeHasher",
            "decompositions": default_decompositions,
            "static_argnums": static_argnums,
        }

        if memory_efficient_fusion is True:
            config["partition_fn"] = min_cut_rematerialization_partition

        return config

    def generate(
        self,
        module,
        param,
        graph_info,
        memory_efficient_fusion,
    ):
        graph_info = deepcopy(graph_info)
        static_argnums = tuple(
            i for i, p in enumerate(param.values()) if not isinstance(p, TensorMeta)
        )

        aot_graph = aot_function(
            module,
            **self.compile_config(
                static_argnums=static_argnums if len(static_argnums) > 0 else None,
                memory_efficient_fusion=memory_efficient_fusion,
            ),
        )

        graph_info["params"] = param
        graph_info["graph"] = aot_graph

        if dist.is_initialized():
            dist.barrier()

        GLOBAL_GRAPH_STORAGE.append(graph_info)
