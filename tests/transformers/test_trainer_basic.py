from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch
from datasets import load_dataset

from oslo.transformers.training_args import TrainingArguments as ota
from oslo.transformers.trainer import Trainer as OTrainer

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

optim = torch.optim.Adam(params=model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optim, T_0=1)

batch_size = 16
datasets = load_dataset("squad")
train_dataset = datasets['train']
valid_dataset = datasets['validation']

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

trainer = OTrainer(
    model=model,
    tokenizer=tokenizer,
    optimizers=(optim, scheduler),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
