# Copyright 2021 TUNiB Inc.

import torch.distributed as dist
from transformers import GPT2Tokenizer

from oslo import GPTNeoForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

model_3d = GPTNeoForCausalLM.from_pretrained_with_parallel(
    "save/3d",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
)
output = tokenizer.batch_decode(
    model_3d.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)

model_3d.save_pretrained_with_parallel("save/3d/merge", save_with_merging=True)
model_1d = GPTNeoForCausalLM.from_pretrained_with_parallel(
    "save/3d/merge", tensor_parallel_size=1
).cuda()

output = tokenizer.batch_decode(
    model_1d.generate(
        **tokenizer("Hello I am Kevin. Today,", return_tensors="pt").to("cuda"),
        num_beams=5
    )
)

if dist.get_rank() == 0:
    print(output)
