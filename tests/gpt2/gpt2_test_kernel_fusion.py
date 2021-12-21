# Copyright 2021 TUNiB Inc.
import os
from time import time

import torch.distributed as dist
from transformers import AutoTokenizer

from oslo.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class TestKernelFusion:
    def test_kernel_fusion(self, input_text):
        """
        before fusion: 3.5706350803375244 sec
        after fusion: 3.2512128353118896 sec
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

        # warm up
        for _ in range(5):
            model.generate(input_ids)

        start = time()
        out1 = model.generate(
            input_ids,
            max_length=200,
            min_length=200,
            num_beams=5,
            no_repeat_ngram_size=4,
        )
        before_fusion = time() - start

        model = model.fuse()

        # warm up
        for _ in range(5):
            model.generate(input_ids)

        start = time()
        out2 = model.generate(
            input_ids,
            max_length=200,
            min_length=200,
            num_beams=5,
            no_repeat_ngram_size=4,
        )
        after_fusion = time() - start

        print("TEST_FUSED_KERNEL_WITHOUT_PARALLEL")
        print(f"\n {tokenizer.batch_decode(out1)}")
        print(f"before fusion: {before_fusion} sec")
        print(f"\n {tokenizer.batch_decode(out2)}")
        print(f"after fusion: {after_fusion} sec")
        print("\n\n")

        del model

    def test_kernel_fusion_with_parallel(self, input_text):
        """
        before fusion: 4.304628610610962 sec
        after fusion: 3.859266519546509 sec
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained_with_parallel(
            "gpt2", tensor_parallel_size=4
        )

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

        # warm up
        for _ in range(5):
            model.generate(input_ids)

        start = time()
        out1 = model.generate(
            input_ids,
            max_length=200,
            min_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        before_fusion = time() - start

        model = model.fuse()

        # warm up
        for _ in range(5):
            model.generate(input_ids)

        start = time()
        out2 = model.generate(
            input_ids,
            max_length=200,
            min_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        after_fusion = time() - start

        if dist.get_rank() == 0:
            print("TEST_FUSED_KERNEL_WITH_PARALLEL")
            print(f"\n {tokenizer.batch_decode(out1)}")
            print(f"before fusion: {before_fusion} sec")
            print(f"\n {tokenizer.batch_decode(out2)}")
            print(f"after fusion: {after_fusion} sec")
            print("\n\n")

        del model


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    test = TestKernelFusion()

    test.test_kernel_fusion_with_parallel("Hello I am Kevin.")
    if dist.get_rank() == 0:
        test.test_kernel_fusion("Hello I am Kevin.")
