from typing import List, Dict, Optional

from utils import BaseProcessor
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class ProcessorForCausalLM(BaseProcessor):
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


class DataCollatorForCausalLM:
    """
    Processing training examples to mini-batch for Gpt2 (clm).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        examples = [example["input_ids"] for example in examples]
        batch = {
            "input_ids": _torch_collate_batch(
                examples, tokenizer=self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
        }
        batch["labels"] = batch["input_ids"].clone()
        return batch