from time import time

import torch
from transformers import GPT2LMHeadModel

import oslo

non_oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().train()
oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().train()

oslo_model = oslo.initialize(
    oslo_model,
    config={
        "kernel_fusion": {
            "enable": True,
            "memory_efficient_fusion": True,
        },
    },
)

BATCH_SIZE, SEQ_LEN = 256, 16
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN, requires_grad=True).long().cuda()

with torch.set_grad_enabled(True):
    # warm up for non-oslo
    for _ in range(10):
        loss = non_oslo_model(input_tensor, labels=input_tensor).loss
        loss.backward()

    start = time()
    for _ in range(10):
        loss = non_oslo_model(input_tensor, labels=input_tensor).loss
        loss.backward()
    print(f"non-oslo: {time() - start}")
    # non-oslo: 1.4633362293243408

    # warm up for oslo
    for _ in range(10):
        loss = oslo_model(input_tensor, labels=input_tensor).loss
        loss.backward()

    start = time()
    for _ in range(10):
        loss = oslo_model(input_tensor, labels=input_tensor).loss
        loss.backward()
    print(f"oslo: {time() - start}")
    # oslo: 1.1438078880310059
