import torch

from oslo.pytorch.kernel_fusion.cuda import CUDA


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
