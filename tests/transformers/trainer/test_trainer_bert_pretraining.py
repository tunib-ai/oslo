from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_bert_pretraining import (
    ProcessorForBertPretraining,
    DataCollatorForBertPretraining,
)
from oslo.transformers.models.bert.modeling_bert import BertForPreTraining
from datasets import load_dataset
import torch
import os


os.environ["WANDB_PROJECT"] = "test_trainer"

dataset = load_dataset("glue", "sst2")
dataset = dataset.rename_column("sentence", "text")

processor = ProcessorForBertPretraining("bert-base-cased")

processed_dataset = dataset.map(
    processor, batched=True, remove_columns=dataset["train"].column_names
)
processed_dataset.cleanup_cache_files()

model = BertForPreTraining.from_pretrained("bert-base-cased")

data_collator = DataCollatorForBertPretraining(processor)

optim = torch.optim.Adam(params=model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=1)

train_dataset = processed_dataset["train"]
valid_dataset = processed_dataset["validation"]

args = OTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    seed=0,
    load_best_model_at_end=True,
    save_steps=100000,
    run_name="test_bert_pretraining",
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
