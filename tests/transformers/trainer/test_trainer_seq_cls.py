from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
from oslo.transformers.models.bert.modeling_bert import BertForSequenceClassification
from datasets import load_dataset
import torch
import os


os.environ["WANDB_PROJECT"] = "test_trainer"

dataset = load_dataset("glue", "sst2")
dataset = dataset.rename_column("sentence", "text")
dataset = dataset.rename_column("label", "labels")

processor = ProcessorForSequenceClassification("bert-base-uncased", 256)
processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()

data_collator = DataCollatorForSequenceClassification(processor)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

optim = torch.optim.Adam(params=model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=1)

train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

args = OTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
    save_steps=100000,
    run_name="test_seq_cls_bert",
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
