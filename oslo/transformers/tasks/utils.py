from pathlib import Path
from typing import Generator, Union

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

SENT_TEXT_SCRIPT = str((Path(__file__).parent / "loading" / "sent_text.py").resolve().absolute())


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):  
    return (dataset[i : i + batch_size][key]
            for i in range(0, len(dataset), batch_size))


def load_corpora(dir_path, corpus_type="docu_json"):
    corpora_dir = Path(dir_path).absolute()
    print(corpora_dir)
    extension = corpus_type.split("_")[-1]

    if extension == "json":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.json")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.json")
    elif extension == "text":
        list_of_file_paths = [str(file_path) for file_path in corpora_dir.rglob("*.txt")]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.txt")
    else:
        raise Exception(f"{extension} is not supported.")

    if corpus_type == "docu_text":
        return load_dataset("text", data_files=list_of_file_paths, split="train")
    elif corpus_type == "docu_json":
        return load_dataset("json", data_files=list_of_file_paths, split="train")
    elif corpus_type == "sent_text":
        return load_dataset(SENT_TEXT_SCRIPT, data_files=list_of_file_paths, split="train")
    elif corpus_type == "sent_json":
        raise NotImplementedError("sent_json will be supported soon.")
    else:
        raise ValueError(f"{corpus_type} must be one of ['docu_text', 'docu_json', 'sent_text', 'sent_json']")


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