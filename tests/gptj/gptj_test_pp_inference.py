# Copyright 2021 TUNiB Inc.

import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo.models.gptj.modeling_gptj import (
    GPTJForCausalLM,
    GPTJForSequenceClassification,
    GPTJModel,
)


class TestPPInference:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.tokenizer = GPT2Tokenizer.from_pretrained("anton-l/gpt-j-tiny-random")

    @torch.no_grad()
    def test_gptj_model(self, fp16):
        model_pp = GPTJModel.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            pipeline_parallel_size=self.num_gpus,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTJModel.from_pretrained_with_parallel("anton-l/gpt-j-tiny-random")
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTJModel.from_pretrained_with_parallel("anton-l/gpt-j-tiny-random")
                .eval()
                .cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        hidden_pp = [_.last_hidden_state for _ in model_pp(**batch_encoding)][0]
        hidden_1d = model_1d(**batch_encoding).last_hidden_state

        if dist.get_rank() == 0:
            print(
                f"\n{TestPPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: {torch.isclose(hidden_1d[0], hidden_pp[0], rtol=1e-2)}\n"
            )

        del model_pp
        del model_1d

    @torch.no_grad()
    def test_gptj_lm_head_model(self, fp16):
        model_pp = GPTJForCausalLM.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            pipeline_parallel_size=self.num_gpus,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTJForCausalLM.from_pretrained_with_parallel(
                    "anton-l/gpt-j-tiny-random"
                )
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTJForCausalLM.from_pretrained_with_parallel(
                    "anton-l/gpt-j-tiny-random"
                )
                .eval()
                .cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        output_pp = model_pp.generate(
            **batch_encoding, num_beams=4, no_repeat_ngram_size=3
        )
        output_1d = model_1d.generate(
            **batch_encoding, num_beams=4, no_repeat_ngram_size=3
        )

        if dist.get_rank() == 0:
            print(
                f"\n{TestPPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{self.tokenizer.decode(output_1d[0])}\n2D:{self.tokenizer.decode(output_pp[0])}\n"
            )

        del model_pp
        del model_1d

    @torch.no_grad()
    def test_gptj_for_classification(self, fp16):
        model_pp = GPTJForSequenceClassification.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            pipeline_parallel_size=self.num_gpus,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTJForSequenceClassification.from_pretrained(
                    "anton-l/gpt-j-tiny-random"
                )
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTJForSequenceClassification.from_pretrained(
                    "anton-l/gpt-j-tiny-random"
                )
                .eval()
                .cuda()
            )

        model_1d.config.pad_token_id = self.tokenizer.eos_token_id
        model_pp.config.pad_token_id = self.tokenizer.eos_token_id

        batch_encoding = self.tokenizer(
            text=["I love you !", "I hate you !"], return_tensors="pt"
        ).to("cuda")

        output_pp = torch.cat(
            [_.logits.argmax(-1) for _ in model_pp(**batch_encoding)], dim=0
        )
        output_1d = model_1d(**batch_encoding).logits.argmax(-1)

        if dist.get_rank() == 0:
            print(
                f"\n{TestPPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{output_1d}\n2D:{output_pp}\n"
            )

        del model_1d
        del model_pp


if __name__ == "__main__":
    test = TestPPInference(num_gpus=4)

    for fp16 in [False, True]:
        test.test_gptj_model(fp16=fp16)
    for fp16 in [False, True]:
        test.test_gptj_lm_head_model(fp16=fp16)
    for fp16 in [False, True]:
        test.test_gptj_for_classification(fp16=fp16)
