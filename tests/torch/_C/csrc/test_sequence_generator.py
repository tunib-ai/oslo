# Originally from Facebook

import math
import unittest

import numpy as np
import torch


from oslo.torch.nn import NGramRepeatBlock

DEFAULT_TEST_VOCAB_SIZE = 100


JIT_MSG = "Targeting OSS scriptability for the 1.6 release"


class TestSequenceGeneratorBase(unittest.TestCase):
    def assertHypoTokens(self, hypo, tokens):
        self.assertTensorEqual(hypo["tokens"], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.0):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo["positional_scores"], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo["tokens"].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo["score"]), 1e-6)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "")
class TestRepeatNgramBlocking(TestSequenceGeneratorBase):
    def test_finds_repetitive_tokens(self):
        bsz, vocab_size, beam_size, step = 2, 4, 1, 3
        generated_tok = torch.tensor(
            [[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.long, device="cuda"
        )
        lprobs = torch.zeros((beam_size * bsz, vocab_size), device="cuda")
        desired_result = lprobs.new_tensor(
            [[0.0, 0.0, -math.inf, 0.0], [0.0, 0.0, 0.0, -math.inf]]
        )

        cuda_ext_result, baseline_result = self._compare_cuda_ext_to_default_implem(
            bsz, beam_size, generated_tok, lprobs, step, 2
        )
        self.assertTensorEqual(cuda_ext_result, desired_result)
        self.assertTensorEqual(baseline_result, desired_result)

    @unittest.skipIf(torch.__version__ < "1.6.0", JIT_MSG)
    def test_jit_no_extension(self):
        bsz, vocab_size, beam_size, step = 2, 4, 1, 3
        generated_tok = torch.tensor(
            [[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.long, device="cuda"
        )
        lprobs = torch.zeros((beam_size * bsz, vocab_size), device="cuda")
        blocker = NGramRepeatBlock(2, use_extension=False)
        base_result = blocker(generated_tok, lprobs.clone(), bsz, beam_size, step)
        scripted_blocker = torch.jit.script(blocker)
        jit_result = scripted_blocker(
            generated_tok, lprobs.clone(), bsz, beam_size, step
        )
        self.assertTensorEqual(base_result, jit_result)

    def test_ngram_blocking_same_as_default_implem(self):
        """Test that cuda extension returns same things as default impl in many settings."""
        vocab_size = 4
        step = 6
        for _ in range(2):
            block_param = np.random.choice([1, 2, 3, 4])
            batch_size = np.random.randint(1, 8)
            beam_size = np.random.choice([1, 2, 4, 8])
            lprobs = torch.zeros((beam_size * batch_size, vocab_size), device="cuda")

            generated_tok = torch.tensor(
                np.random.randint(
                    0, vocab_size, size=(batch_size * beam_size, step + 1)
                ),
                device="cuda",
                dtype=torch.long,
            )
            self._compare_cuda_ext_to_default_implem(
                batch_size,
                beam_size,
                generated_tok,
                lprobs,
                step,
                block_param,
            )

    def _compare_cuda_ext_to_default_implem(
        self, bsz, beam_size, generated_tok, lprobs, step, block_param
    ):
        """Assert that cuda extension and default implem return the same thing."""
        blocker = NGramRepeatBlock(block_param)
        assert blocker.use_extension, "Extension not compiled"
        cuda_ext_result = blocker(
            generated_tok,
            lprobs.clone(),
            bsz,
            beam_size,
            step,
        )
        blocker.use_extension = False
        baseline_result = blocker(
            generated_tok,
            lprobs.clone(),
            bsz,
            beam_size,
            step,
        )
        self.assertTensorEqual(cuda_ext_result, baseline_result)
        blocker.use_extension = True
        return cuda_ext_result, baseline_result


if __name__ == "__main__":
    unittest.main()
