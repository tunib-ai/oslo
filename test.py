
from oslo.transformers.training_args import TrainingArguments as ota
from oslo.transformers.trainer import Trainer
from transformers import TrainingArguments as ta
from transformers import BertTokenizer

import torch

#
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# print(type(tokenizer))
# # ota()
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



# trainer = Trainer()