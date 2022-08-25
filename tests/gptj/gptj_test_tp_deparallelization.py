# Copyright 2021 TUNiB Inc.

import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo import GPTJForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")

model_tp = GPTJForCausalLM.from_pretrained_with_parallel(
    "save/tp", tensor_parallel_size=4
)

output = tokenizer.batch_decode(
    model_tp.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)

model_tp.save_pretrained_with_parallel("save/tp/merge", save_with_merging=True)

model_1d = GPTJForCausalLM.from_pretrained_with_parallel(
    "save/tp/merge", tensor_parallel_size=1
).cuda()

output = tokenizer.batch_decode(
    model_1d.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)
