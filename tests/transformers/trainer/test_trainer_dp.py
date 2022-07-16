import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

from oslo.transformers.training_args import TrainingArguments
from oslo.transformers.trainer import Trainer
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)

oslo_init_dict_form = {
    "data_parallelism": {
        "stage": "zero2",
        "data_parallel_size": 2,
        "sequence_parallel_size": 1
    },
    "model_parallelism": {
        "expert_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "tensor_parallel_depth": 1,
        "tensor_parallel_mode": "tensor_1d"
    },
    "activation_checkpointing": {
        "partitioned_checkpointing": False,
        "contiguous_checkpointing": False
    },
    "kernel_fusion": {
        "memory_efficient_fusion": False
    },
    "lazy_initialization": False,
    "backend": "nccl"
}


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

optim = torch.optim.Adam(params=model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optim, T_0=1)

dataset = load_dataset("glue", 'cola')
dataset = dataset.rename_column("sentence", "text")
dataset = dataset.rename_column("label", "labels")

processor = ProcessorForSequenceClassification("bert-base-uncased", 512)
if processor._tokenizer.pad_token is None:
    processor._tokenizer.pad_token = processor._tokenizer.eos_token

processed_dataset = dataset.map(processor,
                                batched=True,
                                remove_columns=dataset["train"].column_names)
processed_dataset.cleanup_cache_files()
train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

data_collator = DataCollatorForSequenceClassification(processor)

args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    oslo_user_config=oslo_init_dict_form
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    optimizers=(optim, scheduler),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()
