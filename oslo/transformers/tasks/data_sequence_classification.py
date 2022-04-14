from typing import Any, Dict, List, Optional

from transformers import GPT2ForSequenceClassification
from transformers.file_utils import PaddingStrategy
from data_utils import BaseProcessor


class ProcessorForSequenceClassfication(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: Optional[int] = None) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
    
    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
            list_of_str,
            verbose=False,
        )

        return dict_of_training_examples


class DataCollatorForSequenceClassification:
    """
    Processing training examples to mini-batch for Gpt2 (sequence classification).
    """

    def __init__(
        self,
        tokenizer: ProcessorForSequenceClassfication,
        pad_to_multiple_of: Optional[int] = None,
        padding: PaddingStrategy = "longest",
        # model: GPT2ForSequenceClassification = None,
    ):
        self.tokenizer = tokenizer._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.tokenizer._pad_token = self.tokenizer._eos_token
        self.tokenizer.padding_side = "left"
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

