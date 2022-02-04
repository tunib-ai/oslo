import torch
from oslo.pytorch.kernel_fusion.utils.model_output import register_model_output_classes
from oslo.pytorch.kernel_fusion.utils.aot_autograd import aot_module
from oslo.pytorch.kernel_fusion.utils.compilers import (
    ts_compile,
    default_decompositions,
)
from oslo.pytorch.kernel_fusion.utils.partitioners import (
    partition_with_recompute_fwd_in_bwd,
)


class KernelFusionEngine(object):
    def __init__(self, memory_efficient_fusion, model):
        self.memory_efficient_fusion = memory_efficient_fusion
        self.model = model

    @staticmethod
    def _set_jit_fusion_options():
        """Set PyTorch JIT layer fusion options."""
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
            # nvfuser
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # legacy pytorch fuser
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)

    def fuse(self):
        self._set_jit_fusion_options()
        register_model_output_classes(self.model)

        config = {
            "fw_compiler": ts_compile,
            "bw_compiler": ts_compile,
            "hasher_type": "StaticShapheHasher",
            "decompositions": default_decompositions,
        }

        if self.memory_efficient_fusion is True:
            config["partition_fn"] = partition_with_recompute_fwd_in_bwd

        return aot_module(self.model, **config)
