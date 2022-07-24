from abc import ABC, abstractmethod
from typing import Dict, List

from datasets.arrow_dataset import Batch

try:
    from transformers import AutoTokenizer
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


PARALLEL_KEY = {
    "clm": ("input_ids", "attention_mask"),
    "mlm": ("input_ids", "attention_mask"),
    "seq_cls": ("input_ids", "attention_mask"),
    "token_cls": ("input_ids", "attention_mask"),
    "summarization": ("input_ids", "attention_mask"),
    "bert": ("input_ids", "token_type_ids", "attention_mask"),
    "albert": ("input_ids", "token_type_ids", "attention_mask"),
    "bart": (
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ),
    "t5": (
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ),
}


class BaseProcessor(ABC):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._max_length = max_length
        self._chunk_size = max_length
        self._buffer = []

    def save_tokenizer(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)

    @abstractmethod
    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        pass
