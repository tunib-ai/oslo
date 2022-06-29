from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datasets import load_dataset

from oslo.transformers.oslo_init import OsloTrainerConfig


user_config = OsloTrainerConfig("oslo_user_config.json")
print(user_config)