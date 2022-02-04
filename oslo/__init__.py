# Copyright 2021 TUNiB inc.
import torch
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from oslo.pytorch.kernel_fusion.kernel_fusion_engine import KernelFusionEngine

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
engine = KernelFusionEngine(model=model, memory_efficient_fusion=True)
fused_model = engine.fuse()

inputs = tokenizer("hello I am", return_tensors="pt").to("cuda")["input_ids"]

with torch.jit.fuser("fuser2"):
    output = fused_model.generate(input_ids=inputs)

print(output)
