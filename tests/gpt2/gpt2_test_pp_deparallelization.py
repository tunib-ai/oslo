# Copyright 2021 TUNiB Inc.

import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo import GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_pp = GPT2LMHeadModel.from_pretrained_with_parallel(
    "save/pp", pipeline_parallel_size=4
)
output = tokenizer.batch_decode(
    model_pp.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)

model_pp.save_pretrained_with_parallel("save/pp/merge", save_with_merging=True)
model_1d = GPT2LMHeadModel.from_pretrained_with_parallel(
    "save/pp/merge", pipeline_parallel_size=1
).cuda()

output = tokenizer.batch_decode(
    model_1d.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)
