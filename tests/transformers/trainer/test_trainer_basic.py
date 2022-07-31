import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

from oslo.transformers.training_args import TrainingArguments
from oslo.transformers.trainer import Trainer
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("glue", "cola")
dataset = dataset.rename_column("sentence", "text")
dataset = dataset.rename_column("label", "labels")

processor = ProcessorForSequenceClassification("bert-base-uncased", 512)
if processor._tokenizer.pad_token is None:
    processor._tokenizer.pad_token = processor._tokenizer.eos_token

processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()
train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

data_collator = DataCollatorForSequenceClassification(processor)


args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    optim="adam",
    lr_scheduler_type="linear",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()
