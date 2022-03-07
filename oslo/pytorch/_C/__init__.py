import os
import subprocess
import sys
from pathlib import Path

from torch.utils import cpp_extension

DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "torch_extensions",
    "oslo",
)


class Binder(object):
    def __init__(self):
        self.compat = self.get_compatibility_version()

    def base_package(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def includes(self):
        return []

    def base_path(self):
        return Path(self.base_package().__file__).parent.absolute()

    def sources(self):
        return []

    @staticmethod
    def _search_compatibility_version():
        device_query = os.path.join(
            cpp_extension.CUDA_HOME,
            "extras",
            "demo_suite",
            "deviceQuery",
        )

        output = subprocess.check_output(
            [device_query],
            universal_newlines=True,
        ).split("\n")

        versions = []
        for line in output:
            if "CUDA Capability" in line:
                versions.append(line)

        return versions[0].replace(".", "")[-2:].strip()

    @staticmethod
    def _constant_compatibility_version():
        try:
            output = subprocess.check_output(
                [os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc"), "-V"],
                universal_newlines=True,
            )
            cuda_version = output.split()
            cuda_bare_metal_version = cuda_version[
                cuda_version.index("release") + 1
            ].split(".")[0]

            if int(cuda_bare_metal_version) >= 11:
                return 80  # A100
            else:
                return 70  # V100
        except:  # noqa: E722
            return 0

    def get_compatibility_version(self):
        try:
            return self._search_compatibility_version()
        except Exception:
            return self._constant_compatibility_version()

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
        name = self.name()
        ext_path = os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH)
        os.makedirs(os.path.join(ext_path, name), exist_ok=True)

        extra_include_paths = [
            os.path.join(self.base_path(), include) for include in self.includes()
        ]
        sources = [os.path.join(self.base_path(), path) for path in self.sources()]

        op_module = cpp_extension.load(
            name=name,
            sources=sources,
            extra_include_paths=extra_include_paths,
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


class OSLOBinder(Binder):
    def base_package(self):
        from oslo.pytorch._C import csrc

        return csrc

    def name(self):
        return "oslo"

    def includes(self):
        return ["includes"]


class CompilingBinder(OSLOBinder):
    def name(self):
        return "compiling"

    def sources(self):
        return ["CompileCache.cpp"]


class CUDABinder(OSLOBinder):
    def name(self):
        return "cuda"

    def sources(self):
        return ["FusedLayerNorm.cu", "FusedNoRepeatNGram.cu", "CUDABinder.cpp"]
