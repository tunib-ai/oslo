import random
from typing import Any, Dict, List, Optional

from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"): # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


class DataCollatorForGpt2:
    """
    Processing training examples to mini-batch for Gpt2 (clm).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenzier = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, example):
        examples = [example["input_ids"] for example in examples]
        batch = {
            "input_ids": _torch_collate_batch(
                example, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        }
        batch["labels"] = batch["input_ids"].clone()
        return batch

