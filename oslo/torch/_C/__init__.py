import os
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension
from oslo.torch.jit._utils import _set_jit_fusion_options

_SOFTMAX_KERNEL = None


def get_softmax_kernel():
    global _SOFTMAX_KERNEL

    try:
        if _SOFTMAX_KERNEL is None:
            _set_jit_fusion_options()
            _SOFTMAX_KERNEL = SoftmaxBinder().bind()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _SOFTMAX_KERNEL


DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch_extensions",
    "oslo",
)


class Binder(object):
    def __init__(self):
        self.compat = self.get_compatibility_version()

    @property
    def base_path(self):
        from oslo.torch._C import csrc

        return Path(csrc.__file__).parent.absolute()

    @property
    def name(self):
        return "oslo"

    def includes(self):
        return [
            os.path.join(self.base_path, "includes"),
        ]

    def sources(self):
        return []

    @staticmethod
    def get_compatibility_version():
        a, b = torch.cuda.get_device_capability(torch.cuda.current_device())
        return int(str(a) + str(b))

    def bind(self):
        try:
            import ninja
            import pybind11

        except ImportError:
            raise ImportError(
                "Unable to compile C++ code due to ``ninja`` or ``pybind11`` not being installed. "
                "please install them using ``pip install ninja pybind11``."
            )

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH)
        ext_path = os.path.join(ext_path, self.name)
        os.makedirs(ext_path, exist_ok=True)

        op_module = cpp_extension.load(
            name=self.name,
            sources=[os.path.join(self.base_path, path) for path in self.sources()],
            extra_include_paths=self.includes(),
            extra_cflags=self.cxx_args(),
            extra_cuda_cflags=self.nvcc_args(),
            verbose=False,
        )

        return op_module

    @staticmethod
    def cxx_args():
        if sys.platform == "win32":
            return [
                "-O2",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]
        else:
            return [
                "-O3",
                "-std=c++14",
                "-g",
                "-Wno-reorder",
                "-Wno-deprecated",
                "-Wno-deprecated-declarations",
            ]

    def nvcc_args(self, maxrregcount: int = None):
        nvcc_flags = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

        additional_flags = [
            "-gencode",
            f"arch=compute_{self.compat},code=sm_{self.compat}",
        ]

        if maxrregcount:
            additional_flags.append(f"-maxrregcount={maxrregcount}")

        return nvcc_flags + additional_flags


class FusedLayerNormBinder(Binder):
    @property
    def name(self):
        return "oslo_fused_layer_norm"

    def sources(self):
        return ["fused_layer_norm.cu", "FusedLayerNormBinder.cpp"]


class SoftmaxBinder(Binder):
    @property
    def name(self):
        return "oslo_softmax"

    def sources(self):
        return [
            "scaled_masked_softmax.cu",
            "scaled_upper_triang_masked_softmax.cu",
            "SoftmaxBinder.cpp",
        ]
