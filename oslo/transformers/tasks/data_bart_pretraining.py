import random
import numpy as np
from typing import Any, Dict, List, Optional
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor


class ProcessorForBartPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        examples["text"] = self._split_by_full_stops(examples["text"])

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "mask_infilling_labels": []
        }

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )['input_ids']

        for input_ids in list_of_input_ids:
            self._buffer.extend(input_ids)

            mask_span_length = np.random.poisson(lam=3)
            while len(self._buffer) >= self._chunk_size + mask_span_length:
                chunk_ids = self._buffer[: self._chunk_size + mask_span_length]
                mask_chunk_ids = self._prepare_mask_infilling(chunk_ids, mask_span_length)

                dict_of_training_examples["input_ids"].append(mask_chunk_ids)
                dict_of_training_examples["mask_infilling_labels"].append(mask_span_length)

                self._buffer = self._buffer[self._chunk_size + mask_span_length :]

        return dict_of_training_examples
    
    def _split_by_full_stops(self, texts: List[str]) -> List[str]:
        splited_texts = []
        for text in texts:
            splited_sentences = ""
            for sentence in text.split("."):
                if sentence: # 빈 문장은 제거
                    splited_sentences += sentence + "." + self._tokenizer._sep_token
            
            if splited_sentences: # 빈 문서는 제거
                splited_texts.append(splited_sentences)
                
        return splited_texts
    
    def _prepare_mask_infilling(self, chunk_ids: List[int], mask_span_length: int) -> List[int]:
        seq_length = len(chunk_ids)
        start_position, end_position = 0, seq_length
        while self._tokenizer.sep_token_id in chunk_ids[start_position : end_position]:
            end = seq_length - mask_span_length
            start_position = random.randrange(0, end)
            end_position = start_position + mask_span_length
        mask_chunk_ids = chunk_ids[: start_position] + [self._tokenizer.mask_token_id] + chunk_ids[end_position+1 :]
        return mask_chunk_ids


class DataCollatorForBartPretraining:
    """
    Processing training examples to mini-batch for Bart (mask_infilling, sentence_permutation).
    """

    def __init__(
        self,
        processor: ProcessorForBartPretraining,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_sentence_permutaion_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        return batch

    def _prepare_sentence_permutaion_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]

            split_sentences = []
            split_points = filter(lambda x: chunk_ids[x] == self.tokenizer.sep_token_id, range(len(chunk_ids)))
            for split_point in split_points:
                split_point += 1
                if split_sentences:
                    split_sentences.append(chunk_ids[prev_point : split_point])
                else:
                    split_sentences.append(chunk_ids[: split_point])  
                prev_point = split_point
            split_sentences.append(chunk_ids[prev_point :])

            random.shuffle(split_sentences)
            input_ids = []
            for split_sentence in split_sentences:
                input_ids.extend(split_sentence)
            
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            chunk_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "sentence_permutation_labels": chunk_ids,
                    "mask_infilling_labels": example["mask_infilling_labels"],
                }
            )
            
        return output_examples