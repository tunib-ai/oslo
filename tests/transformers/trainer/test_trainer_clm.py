from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
    DataCollatorForCausalLM,
)
from oslo.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from datasets import load_dataset
import torch
import os


os.environ["WANDB_PROJECT"] = "test_trainer"

dataset = load_dataset("glue", "sst2")
dataset = dataset.rename_column("sentence", "text")

processor = ProcessorForCausalLM("gpt2")
if processor._tokenizer.pad_token is None:
    processor._tokenizer.pad_token = processor._tokenizer.eos_token
processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()

model = GPT2LMHeadModel.from_pretrained("gpt2")

data_collator = DataCollatorForCausalLM(processor)

optim = torch.optim.Adam(params=model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=1)

train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

args = OTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
    save_steps=100000,
    run_name="test_gpt2_clm",
)

trainer = OTrainer(
    model=model,
    args=args,
    tokenizer=processor._tokenizer,
    data_collator=data_collator,
    optimizers=(optim, scheduler),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
