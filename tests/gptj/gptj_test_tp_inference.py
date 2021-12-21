# Copyright 2021 TUNiB Inc.

import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo.models.gptj.modeling_gptj import (
    GPTJForCausalLM,
    GPTJForSequenceClassification,
    GPTJModel,
)


class TestTPInference:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.tokenizer = GPT2Tokenizer.from_pretrained("anton-l/gpt-j-tiny-random")

    @torch.no_grad()
    def test_gptj_model(self, fp16):
        model_tp = GPTJModel.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            tensor_parallel_size=self.num_gpus,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTJModel.from_pretrained("anton-l/gpt-j-tiny-random")
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTJModel.from_pretrained("anton-l/gpt-j-tiny-random").eval().cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        output_tp = model_tp(**batch_encoding).last_hidden_state
        output_1d = model_1d(**batch_encoding).last_hidden_state

        if dist.get_rank() == 0:
            print(
                f"\n{TestTPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: {torch.isclose(output_1d[0], output_tp[0], rtol=1e-2)}\n"
            )

        del model_1d
        del model_tp

    @torch.no_grad()
    def test_gptj_lm_head_model(self, fp16):
        model_tp = GPTJForCausalLM.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            tensor_parallel_size=self.num_gpus,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random")
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random")
                .eval()
                .cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        output_tp = model_tp.generate(
            **batch_encoding,
            num_beams=5,
            no_repeat_ngram_size=3,
        )

        output_1d = model_1d.generate(
            **batch_encoding,
            num_beams=5,
            no_repeat_ngram_size=3,
        )

        if dist.get_rank() == 0:
            print(
                f"\n{TestTPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{self.tokenizer.decode(output_1d[0])}\nTP:{self.tokenizer.decode(output_tp[0])}\n"
            )

        del model_1d
        del model_tp

    @torch.no_grad()
    def test_gptj_for_classification(self, fp16):
        model_tp = GPTJForSequenceClassification.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            tensor_parallel_size=self.num_gpus,
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
        model_tp.config.pad_token_id = self.tokenizer.eos_token_id

        batch_encoding = self.tokenizer(
            text=["I love you !", "I hate you !"], return_tensors="pt"
        ).to("cuda")

        output_tp = model_tp(**batch_encoding).logits.argmax(-1)
        output_1d = model_1d(**batch_encoding).logits.argmax(-1)

        if dist.get_rank() == 0:
            print(
                f"\n{TestTPInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{output_1d}\nTP:{output_tp}\n"
            )

        del model_1d
        del model_tp


if __name__ == "__main__":
    test = TestTPInference(num_gpus=4)

    for fp16 in [False, True]:
        test.test_gptj_model(fp16=fp16)

    for fp16 in [False, True]:
        test.test_gptj_lm_head_model(fp16=fp16)

    for fp16 in [False, True]:
        test.test_gptj_for_classification(fp16=fp16)
