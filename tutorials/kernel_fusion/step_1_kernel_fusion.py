from time import time

import torch
from transformers import GPT2LMHeadModel

import oslo

non_oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

oslo_model = oslo.initialize(
    oslo_model,
    config={
        "kernel_fusion": {"enable": True},
    },
)

BATCH_SIZE, SEQ_LEN = 256, 64
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN).long().cuda()

# warm up for non-oslo
for _ in range(10):
    non_oslo_model(input_tensor)

start = time()
for _ in range(10):
    non_oslo_model(input_tensor)
print(f"non-oslo: {time() - start}")
# non-oslo: 1.4633362293243408

# warm up for oslo
for _ in range(10):
    oslo_model(input_tensor)


start = time()
for _ in range(10):
    oslo_model(input_tensor)
print(f"oslo: {time() - start}")
# oslo: 1.1438078880310059
