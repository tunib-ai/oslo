from time import time

import torch
from transformers import T5ForConditionalGeneration

import oslo

non_oslo_model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
oslo_model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()

oslo_model = oslo.initialize(
    oslo_model,
    config={
        "kernel_fusion": {
            "enable": True,
            "custom_cuda_kernels": ["FusedNoRepeatNGram", "FusedRMSNorm"],
        },
    },
)

BATCH_SIZE, SEQ_LEN = 256, 1
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN).long().cuda()

# Warm-up
for _ in range(10):
    non_oslo_model.partition(input_tensor, no_repeat_ngram_size=3)

for _ in range(10):
    oslo_model.partition(input_tensor, no_repeat_ngram_size=3)

# Bench mark
start = time()
for _ in range(10):
    non_oslo_model.partition(input_tensor, no_repeat_ngram_size=3)
print(f"non-oslo: {time() - start}")
# non-oslo: 1.1885042190551758

start = time()
for _ in range(10):
    oslo_model.partition(input_tensor, no_repeat_ngram_size=3)
print(f"oslo: {time() - start}")
# oslo: 0.45142364501953125
