# Copyright 2021 TUNiB Inc.
import inspect
import logging
import os
import subprocess
import sys
from os.path import expanduser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils import cpp_extension
from transformers import LogitsProcessor
from transformers.activations import ACT2FN
from transformers.generation_logits_process import _calc_banned_ngram_tokens

logger = logging.getLogger(__name__)

DEFAULT_TORCH_EXTENSION_PATH = os.path.join(
    expanduser("~"),
    ".cache",
    "torch_extensions",
)

# kernels
_datasets_builder, _custom_kernels = None, None
_datasets_builder_compiling_success = None


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
        torch._C._jit_set_texpr_fuser_enabled(False)  # fuser1
        torch._C._jit_set_nvfuser_enabled(True)  # fuser2
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)


class AbstractKernelBuilder(object):
    def __init__(self):
        self.compat = self.get_compatibility_version()

    @property
    def base_path(self):
        from oslo import fused_kernels

        return Path(fused_kernels.__file__).parent.absolute()

    @property
    def name(self):
        raise NotImplementedError

    def includes(self):
        return []

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
        output = subprocess.check_output(
            [os.path.join(cpp_extension.CUDA_HOME, "bin", "nvcc"), "-V"],
            universal_newlines=True,
        )
        cuda_version = output.split()
        cuda_bare_metal_version = cuda_version[cuda_version.index("release") + 1].split(
            "."
        )[0]

        if int(cuda_bare_metal_version) >= 11:
            return 80  # A100
        else:
            return 70  # V100

    def get_compatibility_version(self):
        try:
            return self._search_compatibility_version()
        except Exception:
            return self._constant_compatibility_version()

    def load(self):
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


class DatasetsBuilder(AbstractKernelBuilder):
    @property
    def name(self):
        return "oslo_datasets"

    def sources(self):
        return [
            os.path.join("bindings", "pybind_datasets.cpp"),
        ]

    def includes(self):
        return [
            os.path.join(self.base_path, "includes"),
        ]


class KernelBuilder(AbstractKernelBuilder):
    @property
    def name(self):
        return "oslo_kernels"

    def sources(self):
        return [
            "fused_ngram_repeat_block.cu",
            "fused_softmax.cu",
            "fused_triang_softmax.cu",
            os.path.join("bindings", "pybind_kernels.cpp"),
        ]

    def includes(self):
        return [
            os.path.join(self.base_path, "includes"),
        ]


def get_custom_kernels():
    global _custom_kernels

    try:
        if _custom_kernels is None:
            _set_jit_fusion_options()
            _custom_kernels = KernelBuilder().load()
    except Exception:
        raise EnvironmentError(
            "Failed compiling custom CUDA kernels. "
            "please check your CUDA environment."
        )

    return _custom_kernels


def get_datasets_builder():
    global _datasets_builder, _datasets_builder_compiling_success

    if _datasets_builder_compiling_success is None:
        try:
            _datasets_builder = DatasetsBuilder().load()
            _datasets_builder_compiling_success = True
            return get_datasets_builder()
        except Exception as e:
            print(
                "Failed to launch C++ dataset builder... using slower python version. "
                f"Error message: {e}"
            )
            _datasets_builder_compiling_success = False
            return get_datasets_builder()

    elif _datasets_builder_compiling_success is True:
        assert _datasets_builder is not None, "C++ dataset builder must not be None."
        return _datasets_builder

    else:
        return None


@torch.jit.script
def bias_dropout(x, bias, prob, training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=prob, training=training)


@torch.jit.script
def bias_dropout_train(x, bias, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return bias_dropout(x, bias, prob, True)


@torch.jit.script
def bias_dropout_inference(x, bias, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return bias_dropout(x, bias, prob, False)


@torch.jit.script
def bias_dropout_residual(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    return F.dropout(x + bias, p=prob, training=training) + residual


@torch.jit.script
def bias_dropout_residual_train(x, bias, residual, prob):
    # type: (Tensor, Tensor,  Tensor, float) -> Tensor
    return bias_dropout_residual(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_residual_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_residual(x, bias, residual, prob, False)


@torch.jit.script
def bias_gelu_fwb(y, bias):
    x = y + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def bias_gelu_bwd(g, y, bias):
    x = y + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g


class ScaledUpperTriangMaskedSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = (
            get_custom_kernels().scaled_upper_triang_masked_softmax_forward(
                inputs, scale_t[0]
            )
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = get_custom_kernels().scaled_upper_triang_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None


class ScaledMaskedSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])

        softmax_results = get_custom_kernels().scaled_masked_softmax_forward(
            inputs, mask, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = get_custom_kernels().scaled_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class NGramRepeatBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size):
        return get_custom_kernels().ngram_repeat_block_forward(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )

    def backward(*args):
        raise NotImplementedError


class FusedBiasGeLU(torch.autograd.Function):
    """
    Kernel fusion function: Bias + GeLU
    """

    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu_fwb(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


class FusedBiasActivation(object):
    """
    Kernel fusion function: Bias + Activation

    Before fusion: MLP FFN1 -> (Bias Addition -> Activation) -> MLP FFN2 -> ...
    After fusion: MLP FFN1 -> Fused Bias Activation -> MLP FFN2 -> ...
    """

    @staticmethod
    def apply(x, bias, act):
        if "gelu" in act:
            return FusedBiasGeLU.apply(x, bias)
        else:
            return ACT2FN[act](x + bias)


class FusedBiasDropout(object):
    """
    Kernel fusion function: Bias + Dropout

    Before fusion: MLP FFN2 -> (Bias Addition -> Dropout) -> Residual Addition ->. ..
    After fusion: MLP FFN2 -> Fused Bias Dropout -> Residual Addition -> ...
    """

    @staticmethod
    def apply(x, bias, training, drop_prop):
        if training:
            return bias_dropout_train(x, bias, drop_prop)
        else:
            return bias_dropout_inference(x, bias, drop_prop)


class FusedBiasDropoutResidual(object):
    """
    Kernel fusion function: Bias + Dropout + Residual

    Before fusion: MLP FFN2 -> (Bias Addition -> Dropout -> Residual Addition) -> LayerNorm -> ...
    After fusion: MLP FFN2 -> Fused Bias Dropout Residual -> LayerNorm -> ...
    """

    @staticmethod
    def apply(x, bias, residual, training, drop_prop):
        if training:
            return bias_dropout_residual_train(x, bias, residual, drop_prop)
        else:
            return bias_dropout_residual_inference(x, bias, residual, drop_prop)


class FusedScaleMaskSoftmax(object):
    """
    Kernel fusion function: Scale + Mask + Softmax

    Before fusion: Matmul (Q * K^T) -> Scale -> Mask -> Softmax -> Matmul (Score * V) -> ...
    After fusion: Matmul (Q * K^T) -> (Scale + Mask + Softmax) -> Matmul (Score * V) -> ...
    """

    @staticmethod
    def apply(input, pad_mask, scale, use_triang_mask):
        bsz, np, sq, sk = input.size()
        scale = scale if scale is not None else 1.0

        if use_triang_mask:
            if pad_mask is not None:
                input = input + pad_mask
            return ScaledUpperTriangMaskedSoftmaxFunction.apply(
                input.view(-1, sq, sk), scale
            ).view(bsz, np, sq, sk)

        if pad_mask is None:
            pad_mask = torch.zeros(1, 1, sq, sk, device=input.device, dtype=input.dtype)
            return ScaledMaskedSoftmaxFunction.apply(input, pad_mask.bool(), scale)

        else:
            return ScaledMaskedSoftmaxFunction.apply(
                input, pad_mask.repeat(1, 1, sq, 1).bool(), scale
            )

    @staticmethod
    def is_available(dtype, bsz, np, sq, sk, use_triang_mask):
        if dtype != torch.half or sk > 2048 or sk <= 0:
            return False

        bsz_per_block = get_custom_kernels().get_batch_per_block(sq, sk, bsz, np)

        if use_triang_mask:
            if (
                sq == sk
                and (sk <= 64 or sk % 4 == 0)
                and (bsz * np) % bsz_per_block == 0
            ):
                return True

        else:
            if sq > 1 and sq % bsz_per_block == 0:
                return True

        return False


class FusedNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int, batch_size: int, beam_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
            )
        self.ngram_size = ngram_size
        self.batch_size = batch_size
        self.beam_size = beam_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]

        if input_ids.is_cuda and scores.is_cuda:
            scores = NGramRepeatBlockFunction.apply(
                input_ids,
                scores.float(),
                self.batch_size,
                cur_len - 1,
                self.beam_size,
                self.ngram_size,
            )

        else:
            banned_batch_tokens = _calc_banned_ngram_tokens(
                self.ngram_size,
                input_ids,
                num_batch_hypotheses,
                cur_len,
            )

            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores


class FusedKernelMixin(object):
    def fuse(self, modules=None):
        user_modules = modules
        # change variable name to prevent mistakes

        assert user_modules is None or isinstance(
            user_modules, list
        ), "Param `modules` must be type of List[Class]."

        if self.is_fusable:
            for policy in self.get_layer_policies():
                _fused_modules = policy.fused_modules()

                if user_modules is not None:
                    for user_module in user_modules:
                        assert user_module in _fused_modules, (
                            f"Module {user_module} can't be fused! "
                            f"{self.__class__.__qualname__} only supports {list(_fused_modules.keys())}"
                        )
                        assert inspect.isclass(user_module), (
                            f"The element in the ``modules`` must be Class type. "
                            f"but you input {type(user_module)}"
                        )

                for model_module in self.modules():
                    if model_module.__class__ in _fused_modules:
                        if (
                            user_modules is not None
                            and model_module.__class__ not in user_modules
                        ):
                            print(model_module.__class__.__qualname__)
                            continue
                            # skip if module not in ``modules`` that user input.
                        else:
                            fused_module = _fused_modules[model_module.__class__]
                            model_module.__class__ = fused_module
                            setattr(model_module, "config", self.config)
        else:
            raise RuntimeError(
                "This model doesn't support kernel fusion. please check the document."
            )

        return self
