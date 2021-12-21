# Copyright 2021 TUNiB Inc.

import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
    GPTNeoModel,
)


class Test3DInference:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    @torch.no_grad()
    def test_gpt_neo_model(self, fp16):
        model_3d = GPTNeoModel.from_pretrained_with_parallel(
            "EleutherAI/gpt-neo-125M",
            tensor_parallel_size=self.num_gpus // 2,
            pipeline_parallel_size=self.num_gpus // 2,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTNeoModel.from_pretrained_with_parallel("EleutherAI/gpt-neo-125M")
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTNeoModel.from_pretrained_with_parallel("EleutherAI/gpt-neo-125M")
                .eval()
                .cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        hidden_3d = [_.last_hidden_state for _ in model_3d(**batch_encoding)][0]
        hidden_1d = model_1d(**batch_encoding).last_hidden_state

        if dist.get_rank() == 0:
            print(
                f"\n{Test3DInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: {torch.isclose(hidden_1d[0], hidden_3d[0], rtol=1e-2)}\n"
            )

        del model_3d
        del model_1d

    @torch.no_grad()
    def test_gpt_neo_lm_head_model(self, fp16):
        model_3d = GPTNeoForCausalLM.from_pretrained_with_parallel(
            "EleutherAI/gpt-neo-125M",
            tensor_parallel_size=self.num_gpus // 2,
            pipeline_parallel_size=self.num_gpus // 2,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTNeoForCausalLM.from_pretrained_with_parallel(
                    "EleutherAI/gpt-neo-125M"
                )
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTNeoForCausalLM.from_pretrained_with_parallel(
                    "EleutherAI/gpt-neo-125M"
                )
                .eval()
                .cuda()
            )

        batch_encoding = self.tokenizer(
            text="Hello I am Kevin. Today,", return_tensors="pt"
        ).to("cuda")

        output_3d = model_3d.generate(
            **batch_encoding, num_beams=4, no_repeat_ngram_size=3
        )
        output_1d = model_1d.generate(
            **batch_encoding, num_beams=4, no_repeat_ngram_size=3
        )

        if dist.get_rank() == 0:
            print(
                f"\n{Test3DInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{self.tokenizer.decode(output_1d[0])}\n2D:{self.tokenizer.decode(output_3d[0])}\n"
            )

        del model_3d
        del model_1d

    @torch.no_grad()
    def test_gpt_neo_for_classification(self, fp16):
        model_3d = GPTNeoForSequenceClassification.from_pretrained_with_parallel(
            "EleutherAI/gpt-neo-125M",
            tensor_parallel_size=self.num_gpus // 2,
            pipeline_parallel_size=self.num_gpus // 2,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        ).eval()

        if fp16:
            model_1d = (
                GPTNeoForSequenceClassification.from_pretrained(
                    "EleutherAI/gpt-neo-125M"
                )
                .half()
                .eval()
                .cuda()
            )
        else:
            model_1d = (
                GPTNeoForSequenceClassification.from_pretrained(
                    "EleutherAI/gpt-neo-125M"
                )
                .eval()
                .cuda()
            )

        model_1d.config.pad_token_id = self.tokenizer.eos_token_id
        model_3d.config.pad_token_id = self.tokenizer.eos_token_id

        batch_encoding = self.tokenizer(
            text=["I love you !", "I hate you !"], return_tensors="pt"
        ).to("cuda")

        output_3d = torch.cat(
            [_.logits.argmax(-1) for _ in model_3d(**batch_encoding)], dim=0
        )
        output_1d = model_1d(**batch_encoding).logits.argmax(-1)

        if dist.get_rank() == 0:
            print(
                f"\n{Test3DInference.__qualname__}:\n"
                f"--fp16:{fp16}\n"
                f"--test result: \n1D:{output_1d}\n2D:{output_3d}\n"
            )

        del model_1d
        del model_3d


if __name__ == "__main__":
    test = Test3DInference(num_gpus=4)

    for fp16 in [False, True]:
        test.test_gpt_neo_model(fp16=fp16)

    for fp16 in [False, True]:
        test.test_gpt_neo_lm_head_model(fp16=fp16)

    for fp16 in [False, True]:
        test.test_gpt_neo_for_classification(fp16=fp16)
