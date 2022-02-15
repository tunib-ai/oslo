from time import time

import torch
from transformers import GPT2LMHeadModel

import oslo

BATCH_SIZE, SEQ_LEN = 256, 16
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN).long().cuda()

non_oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

oslo_model = oslo.initialize(
    oslo_model,
    config={
        "kernel_fusion": {"enable": True},
    },
)


# Warm-up
for _ in range(10):
    non_oslo_model(input_tensor)

for _ in range(10):
    oslo_model(input_tensor)

# Bench mark
start = time()
for _ in range(10):
    non_oslo_model(input_tensor)
print(f"non-oslo: {time() - start}")
# non-oslo: 0.25797200202941895


start = time()
for _ in range(10):
    oslo_model(input_tensor)
print(f"oslo: {time() - start}")
# oslo: 0.20798110961914062
