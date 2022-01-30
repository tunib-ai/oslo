"""
Model parallelism tutorial step 1:
How to use the tensor parallelism for inference?
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

import oslo

# NOTE: This script must be executed with multiprocessing
# I strongly recommend to use torch.distributed.launch like this:
# ``python -m torch.distributed.launch --nproc_per_node 4 step_1_tp_inference.py``

# 1. Create model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Parallelize the model
# - ``tensor_parallel_size`` must be same or smaller than total num of gpus.
# - ``tensor_parallel_size`` must be power of 2. (e.g. 2, 4, 8, 16, ...)
# - ``tensor_parallel_size`` must be positive number.
# If you don't have 4 GPUs, please modify this script like:
# model = oslo.initialize(model, config={"model_parallelism": {"tensor_parallel_size": num_gpus}})
model = oslo.initialize(
    model, config={"model_parallelism": {"tensor_parallel_size": 4}}
)

# NOTE: you can also use json file (oslo-config.json)
# {
#      "model_parallelism": {
#           "tensor_parallel_size": 4
#       }
# }
# And you can do like this:
# model = oslo.initialize(model, config="oslo-config.json")

# 3. Do inference as usual !
# Any other tasks like classification can be performed as usual.
text = "I don't want a lot for Christmas. There is just one thing"
tokens = tokenizer(text, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**tokens, num_beams=3)[0]))
