
from transformers import TrainingArguments as ta
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset

from oslo.transformers.training_args import TrainingArguments as ota
from oslo.transformers.trainer import Trainer as OTrainer


class CustomDataset(Dataset):
    def __init__(self, text, tokenizer, labels=None):
        self.text = text
        self.tokenizer = tokenizer
        self.labels = labels

    def __getitem__(self, idx):
        print(self.text[idx])
        item = self.tokenizer(self.text[idx])
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.text)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


optim = torch.optim.Adam(params=model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=1)

project_path = "/Users/hspark/Projects/_Others/nsmc"
train_data = pd.read_csv(project_path + "/ratings_train.txt", sep='\t')
valid_data = pd.read_csv(project_path + "/ratings_test.txt", sep='\t')

train_data.dropna(subset=["document"], inplace=True)
valid_data.dropna(subset=["document"], inplace=True)


# train_encoded = tokenizer(train_data, padding=True, truncation=True, max_length=512)
# valid_encoded = tokenizer(valid_data, padding=True, truncation=True, max_length=512)

train_dataset = CustomDataset(list(train_data['document']), tokenizer, labels=list(train_data['label']))
valid_dataset = CustomDataset(list(valid_data['document']), tokenizer, labels=list(valid_data['label']))

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

# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     optimizers=(optim, scheduler),
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
# )

trainer = OTrainer(
    model=model,
    tokenizer=tokenizer,
    optimizers=(optim, scheduler),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()

