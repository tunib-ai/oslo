from abc import ABC, abstractmethod
from typing import Dict, List

from transformers import AutoTokenizer


class BaseProcesser(ABC):
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


class GPT2Processer(BaseProcesser):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length
    
    def __call__(self, list_of_str: List[str]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = {}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            list_of_str,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                training_example = self._tokenizer.prepare_for_model(
                    chunk_ids,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )

                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])
                
                self._buffer = self._buffer[self._chunk_size :]
        
        return dict_of_training_examples