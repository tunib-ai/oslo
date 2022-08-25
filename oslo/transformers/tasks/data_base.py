from abc import ABC, abstractmethod
from typing import Dict, List, Optional
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


class ParallelKeys:
    CLM = ["input_ids", "attention_mask"]
    MLM = ["input_ids", "attention_mask"]
    SEQ_CLS = ["input_ids", "token_type_ids", "attention_mask"]
    TOKEN_CLS = ["input_ids", "attention_mask"]
    SUMMARIZATION = [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
    BERT = ["input_ids", "token_type_ids", "attention_mask"]
    ALBERT = ["input_ids", "token_type_ids", "attention_mask"]
    BART = [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
    T5 = [
        "input_ids",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]


def pad_labels(
    labels,
    tokenizer,
    label_pad_token_id: int,
    pad_to_multiple_of: Optional[int] = None,
):
    labels = tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_attention_mask=False,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of,
    )["input_ids"]

    labels.masked_fill_(labels == tokenizer.pad_token_id, label_pad_token_id)
    return labels
