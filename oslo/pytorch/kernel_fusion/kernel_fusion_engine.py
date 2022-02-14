from logging import getLogger

import torch

from oslo.pytorch.kernel_fusion.utils.torch_version import higher_than

logger = getLogger(__name__)


class KernelFusionEngine(object):
    def __init__(
        self,
        model,
        memory_efficient_fusion,
        custom_cuda_kernels=None,
    ):
        if custom_cuda_kernels is None:
            custom_cuda_kernels = []

        self.model = model
        self.custom_cuda_kernels = custom_cuda_kernels
        self.memory_efficient_fusion = memory_efficient_fusion
        self._set_jit_fusion_options()
        self.is_fused = False

    def fuse(self):
        if not self.is_fused:
            if len(self.custom_cuda_kernels) > 0:
                from oslo.pytorch.kernel_fusion.cuda.engine import (
                    CustomCUDAKernelEngine,
                )

                custom_cuda_kernel_engine = CustomCUDAKernelEngine(
                    model=self.model,
                    kernels=self.custom_cuda_kernels,
                )
                custom_cuda_kernel_engine.fuse()

            if self.memory_efficient_fusion is True:
                from oslo.pytorch.kernel_fusion.mem_efficient.engine import (
                    MemoryEfficientFusionEngine,
                )

                mem_efficient_fusion_engine = MemoryEfficientFusionEngine(
                    model=self.model
                )
                self.model = mem_efficient_fusion_engine.fuse()
            else:
                from oslo.pytorch.kernel_fusion.jit_partial.engine import (
                    JITPartialCompilingEngine,
                )

                jit_partial_compiling_engine = JITPartialCompilingEngine(
                    model=self.model
                )
                jit_partial_compiling_engine.fuse()

            self.is_fused = True

        return self.model

    @staticmethod
    def _set_jit_fusion_options():
        """Set PyTorch JIT layer fusion options."""
        if higher_than(1, 10):
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
            return "nv_fuser"
        else:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)
            return "torch_fuser"
