from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)
from oslo.transformers.models.bart.modeling_bart import BartForConditionalGeneration
from datasets import load_dataset
import torch
import os


os.environ["WANDB_PROJECT"] = "test_trainer"

dataset = load_dataset("glue", "sst2")
dataset = dataset.rename_column("sentence", "text")

processor = ProcessorForBartPretraining("facebook/bart-base")

processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

data_collator = DataCollatorForBartPretraining(processor)

optim = torch.optim.Adam(params=model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=1)

train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

args = OTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
    save_steps=100000,
    run_name="test_bart_pretraining",
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
