import os
from copy import deepcopy

import torch
import torch.distributed as dist
from oslo.pytorch._C import DEFAULT_TORCH_EXTENSION_PATH
from oslo.pytorch.kernel_fusion.compile.partitioners import (
    min_cut_rematerialization_partition,
)

from oslo.pytorch.kernel_fusion.compile.aot_autograd import aot_function
from oslo.pytorch.kernel_fusion.compile.compilers import (
    ts_compile,
    default_decompositions,
)
from oslo.pytorch.kernel_fusion.dynamic_shapes import GLOBAL_GRAPH_STORAGE
from oslo.pytorch.kernel_fusion.params import TensorMeta, StaticArgMeta
from oslo.pytorch.kernel_fusion.utils import is_iterable


class AOTManager(object):
    def __init__(self, model, batch_size, seq_len):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len

    def meta2value(self, values):
        if isinstance(values, TensorMeta):
            values = torch.ones(
                values.size,
                device=self.model.device,
                dtype=self.model.dtype if values.dtype is None else values.dtype,
            )
        elif isinstance(values, StaticArgMeta):
            values = values.type()
        elif is_iterable(values):
            if isinstance(values, dict):
                values = {k: self.meta2value(v) for k, v in values.items()}
            else:
                values = [self.meta2value(v) for v in values]

        return values

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
            "hasher_type": "StaticShapheHasher",
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
        graph_info["params"] = "-".join(list(param.keys()))
        static_argnums = tuple(
            i for i, p in enumerate(param.values()) if not isinstance(p, TensorMeta)
        )

        aot_forward = aot_function(
            module.forward,
            **self.compile_config(
                static_argnums=static_argnums if len(static_argnums) > 0 else None,
                memory_efficient_fusion=memory_efficient_fusion,
            ),
        )

        tensor_param = self.meta2value(param)
        aot_forward(**tensor_param)
        graph_info["params"] = param
        graph_info["graph"] = aot_forward
        del tensor_param

        if dist.is_initialized():
            dist.barrier()

        GLOBAL_GRAPH_STORAGE.append(graph_info)
