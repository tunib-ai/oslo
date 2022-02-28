from transformers import GPT2LMHeadModel, GPT2Tokenizer

import oslo

config = {
    "model_parallelism": {
        "enable": True,
        "tensor_parallel_size": 4,
        "deployment_mode": True,
    }
}


model = GPT2LMHeadModel.from_pretrained("gpt2")
model = oslo.initialize(model, config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.generate(**tokenizer("hello", return_tensors="pt").to("cuda"))

while True:
    pass