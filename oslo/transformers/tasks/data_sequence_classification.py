from typing import Any, Dict, List, Optional

from transformers.file_utils import PaddingStrategy
from datasets.arrow_dataset import Batch
from data_base import BaseProcessor


class ProcessorForSequenceClassification(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: Optional[int] = None) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
    
    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert "text" in column_names, "The name of dataset column that you want to tokenize must be 'text'"
        assert "labels" in column_names, "The name of dataset column that you want to use as a label must be 'labels'"
        
        dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
            examples["text"],
            verbose=False,
        )
        dict_of_training_examples["labels"] = examples["labels"]

        return dict_of_training_examples


class DataCollatorForSequenceClassification:
    """
    Processing training examples to mini-batch for Gpt2 (sequence classification).
    """

    def __init__(
        self,
        tokenizer: ProcessorForSequenceClassification,
        pad_to_multiple_of: Optional[int] = None,
        padding: PaddingStrategy = "longest",
    ):
        self.tokenizer = tokenizer._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # assert self.tokenizer._padtoken is not None, ""

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

