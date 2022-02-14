from oslo.pytorch.kernel_fusion.cuda.fused_ngram_blocking import (
    get_ngram_logit_processor,
)
from oslo.pytorch.kernel_fusion.cuda.fused_normalization import FusedRMSNorm


class CustomCUDAKernelEngine(object):
    def supported_kernels(self):
        return {
            "FusedNoRepeatNGram": self.fused_no_repeat_ngram_logits_processor,
            "FusedRMSNorm": self.fused_rms_norm,
        }

    def __init__(self, model, kernels):
        for kernel in kernels:
            if kernel not in self.supported_kernels():
                raise ValueError(
                    f"Unknown kernel name - {kernel}. "
                    f"Currently we support {self.supported_kernels()}."
                )

        self.model = model
        self.kernels = kernels

    def fuse(self):
        for kernel in self.kernels:
            apply_fn = self.supported_kernels()[kernel]
            apply_fn(model=self.model)

    @staticmethod
    def fused_no_repeat_ngram_logits_processor(model):
        from transformers import generation_utils

        orig_generate_fn = model.generate

        def generate(*args, **kwargs):
            num_beams = kwargs.get("num_beams", 1)
            input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else args[0]

            generation_utils.NoRepeatNGramLogitsProcessor = get_ngram_logit_processor(
                num_beams=num_beams, batch_size=input_ids.size(0)
            )

            return orig_generate_fn(*args, **kwargs)

        model.generate = generate

    @staticmethod
    def fused_rms_norm(model):
        from transformers.models.t5.modeling_t5 import T5LayerNorm, T5PreTrainedModel

        if not isinstance(model, T5PreTrainedModel):
            raise ValueError(
                f"FusedRMSNorm is available only for T5 based models. "
                f"but your model is {model.__class__.__qualname__}."
            )

        for module in model.modules():
            if isinstance(module, T5LayerNorm):
                setattr(module, "elementwise_affine", True)
                setattr(module, "normalized_shape", module.weight.size())
                setattr(module, "eps", module.variance_epsilon)
                module.__class__ = FusedRMSNorm
