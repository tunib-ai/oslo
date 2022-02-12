import os
from copy import deepcopy

import torch
import torch.distributed as dist

from oslo.pytorch._C import DEFAULT_TORCH_EXTENSION_PATH
from oslo.pytorch.kernel_fusion.compile.aot_autograd import aot_function
from oslo.pytorch.kernel_fusion.compile.compat import _stateless
from oslo.pytorch.kernel_fusion.compile.compilers import (
    ts_compile,
    default_decompositions,
)
from oslo.pytorch.kernel_fusion.compile.partitioners import (
    min_cut_rematerialization_partition,
)
from oslo.pytorch.kernel_fusion.manage import GLOBAL_GRAPH_STORAGE
from oslo.pytorch.kernel_fusion.params import TensorMeta


class AOTManager(object):
    def __init__(self, model, batch_size, seq_len):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len

    @staticmethod
    def graph_file_name(graph_info):
        file_names = []
        for key, val in graph_info.items():
            if key in ["module_name", "module_cls", "model_cls"] or not isinstance(
                val, str
            ):
                val = f"{key}={val}"
            file_names.append(val)

        os.makedirs(
            os.path.join(DEFAULT_TORCH_EXTENSION_PATH, "oslo", "kernel_fusion"),
            exist_ok=True,
        )
        file_name = ".".join(file_names).lower().replace("compiler", "")
        return os.path.join(DEFAULT_TORCH_EXTENSION_PATH, "oslo", file_name)

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

    @torch.no_grad()
    def compile(
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

        def functional_call(named_params, named_buffers, *args, **kwargs):
            params_and_buffers = {**named_params, **named_buffers}
            return _stateless.functional_call(module, params_and_buffers, args, kwargs)

        aot_graph = aot_function(
            functional_call,
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
