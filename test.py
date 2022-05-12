
from oslo.transformers.training_args import TrainingArguments as ota

from transformers import TrainingArguments as ta

import torch

# ota()
print(ota)

print(ta)

args = ota(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)

