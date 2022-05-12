import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
try:
    from transformers import (
        DataCollatorForWholeWordMask,
        T5Tokenizer,
        T5TokenizerFast,
    )
    from transformers.data.data_collator import _torch_collate_batch
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForT5PrefixLanguageModeling(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "labels": []
        }

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )['input_ids']

        list_of_input_ids, list_of_labels = self._prepare_for_prefix_language_modeling(list_of_input_ids)
        for input_ids, labels in zip(list_of_input_ids, list_of_labels):
            if len(input_ids) <= self._max_length and len(labels) <= self._max_length:
                dict_of_training_examples["input_ids"].append(input_ids)
                dict_of_training_examples["labels"].append(labels)

        return dict_of_training_examples
    
    def _prepare_for_prefix_language_modeling(self, list_of_input_ids: List[List[int]]) -> Tuple[List[List[str]], List[List[str]]]:
        inputs, targets = [], []
        for input_ids in list_of_input_ids:
            seq_length = len(input_ids)
            if seq_length >= 3:
                start, end = seq_length // 3, seq_length // 3 * 2
                split_position = random.randrange(start, end)
                inputs.append(input_ids[:split_position])
                targets.append(input_ids[split_position:])
        
        return inputs, targets


class DataCollatorForT5PrefixLanguageModeling:
    """
    Processing training examples to mini-batch for T5 (prefix language modeling).
    """

    def __init__(
        self,
        processor: ProcessorForT5PrefixLanguageModeling,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]

        batch = {
            "input_ids": _torch_collate_batch(
                input_ids,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
            "labels": _torch_collate_batch(
                labels,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
        }
        return batch