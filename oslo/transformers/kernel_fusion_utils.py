import torch


class NGramRepeatBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size):
        return CUDA.ngram_repeat_block_forward(
            tokens, lprobs, bsz, step, beam_size, no_repeat_ngram_size
        )

    def backward(*args):
        raise NotImplementedError


def get_ngram_logit_processor(batch_size, num_beams):
    from transformers import LogitsProcessor
    from transformers.generation_logits_process import _calc_banned_ngram_tokens

    class FusedNoRepeatNGramLogitsProcessor(LogitsProcessor):
        def __init__(self, ngram_size: int):
            if not isinstance(ngram_size, int) or ngram_size <= 0:
                raise ValueError(
                    f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}"
                )
            self.ngram_size = ngram_size

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
                    batch_size,
                    cur_len - 1,
                    num_beams,
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

    return FusedNoRepeatNGramLogitsProcessor


def fused_no_repeat_ngram_logits_processor(model):
    from transformers import generation_utils

    orig_generate_fn = model.generate

    def generate(*args, **kwargs):
        num_beams = kwargs.get("num_beams", 1)
        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else args[0]

        generation_utils.NoRepeatNGramLogitsProcessor = get_ngram_logit_processor(
            num_beams=num_beams, batch_size=input_ids.size(0)
        )
        print("my code")

        return orig_generate_fn(*args, **kwargs)

    model.generate = generate


def fused_rms_norm(model):
    print(model)
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
    print(model)
