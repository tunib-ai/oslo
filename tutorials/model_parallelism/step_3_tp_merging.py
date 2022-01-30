"""
Model parallelism tutorial step 3:
How to merge the parallelized checkpoints?
"""

from transformers import AutoModelForCausalLM

import oslo

# NOTE: This script must be executed with multiprocessing
# I strongly recommend to use torch.distributed.launch like this:
# ``python -m torch.distributed.launch --nproc_per_node 4 step_2_tp_training.py``

# 1. Create model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Parallelize the model
# - ``tensor_parallel_size`` must be same or smaller than total num of gpus.
# - ``tensor_parallel_size`` must be power of 2. (e.g. 2, 4, 8, 16, ...)
# - ``tensor_parallel_size`` must be positive number.
model = oslo.initialize(
    model, config={"model_parallelism": {"tensor_parallel_size": 4}}
)

# 3. Load parallelized checkpoints
# We support the method ``from_parallelized``.
# Input parallelized checkpoint path to here.
model = model.from_parallelized("./parallel_ckpt")

# 4. Merge parallelized checkpoints
# You must set the argument ``merge_checkpoints`` as True.
# This is a bit slow, so we recommend to use this after training.
model.save_parallelized("./merged_ckpt", merge_checkpoints=True)
