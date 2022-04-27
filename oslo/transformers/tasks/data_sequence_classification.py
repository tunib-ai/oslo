from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from transformers import GPT2ForSequenceClassification, AutoTokenizer
from transformers.file_utils import PaddingStrategy
from datasets.arrow_dataset import Batch


class BaseProcessor(ABC):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._chunk_size = max_length
        self._buffer = []

    def save_tokenizer(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)

    @abstractmethod
    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        pass


class ProcessorForSequenceClassification(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: Optional[int] = None) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
    
    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
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
        # model: GPT2ForSequenceClassification = None,
    ):
        self.tokenizer = tokenizer._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        if self.tokenizer._pad_token is None:
            self.tokenizer._pad_token = self.tokenizer._eos_token
        # self.tokenizer.padding_side = "left"
        # model.config.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
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

