from abc import ABC, abstractmethod
from typing import Dict, List

from datasets.arrow_dataset import Batch

try:
    from transformers import AutoTokenizer
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


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
