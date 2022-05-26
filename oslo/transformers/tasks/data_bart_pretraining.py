import random
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import warnings
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
try:
    from transformers import (
        BartTokenizer,
        BartTokenizerFast,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForBartPretraining(BaseProcessor):
    def __init__(
        self,
        model_name_or_path: str, 
        max_length: int, 
        ) -> None:
        super().__init__(model_name_or_path, max_length)

        if not isinstance(self._tokenizer, (BartTokenizer, BartTokenizerFast)):
            warnings.warn(
                "PorcessorForBartPretraining is only suitable for BartTokenizer-like tokenizers."
            )

        self._chunk_size = max_length - 1

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {"input_ids": []}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )["input_ids"]

        for input_ids in list_of_input_ids:
            input_ids += [self._tokenizer.sep_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples
    

class DataCollatorForBartPretraining:
    """
    Processing training examples to mini-batch for Bart (text_infilling, sentence_permutation).
    """

    def __init__(
        self,
        processor: ProcessorForBartPretraining,
        mlm_probability: float = 0.15, 
        possion_lambda: float = 3.0,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.possion_lambda = possion_lambda
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_token_id = processor._tokenizer.mask_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_noise_text_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        return batch

    def _prepare_noise_text_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            labels = chunk_ids[:]

            chunk_ids = self.replace_span(chunk_ids)
            chunk_ids = self.sentence_permutation(chunk_ids)
            
            chunk_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)
            labels = self.tokenizer.build_inputs_with_special_tokens(labels)[1:]

            output_examples.append(
                {
                    "input_ids": chunk_ids,
                    "labels": labels,
                }
            )
            
        return output_examples

    def replace_span(self, input_ids):
        length = len(input_ids)
        num_noise_tokens = int(np.round(length * self.mlm_probability))

        # pick the lengths of the noise spans
        def _possion_segmentation(num_noise_tokens):
            segment_lengths = []
            while sum(segment_lengths) < num_noise_tokens:
                span_length = np.random.poisson(lam=self.possion_lambda)
                segment_lengths.append(span_length)

            return segment_lengths

        noise_span_lengths = _possion_segmentation(num_noise_tokens)

        for noise_span_length in noise_span_lengths:
            max_idx = len(input_ids) - noise_span_length
            start_idx = np.random.choice(np.arange(max_idx))
            while self.mask_token_id in input_ids[start_idx : start_idx+noise_span_length]:
                start_idx = np.random.choice(np.arange(max_idx))
            input_ids = input_ids[ : start_idx] + [self.mask_token_id] + input_ids[start_idx+noise_span_length : ]

        return input_ids
    
    def sentence_permutation(self, input_ids):
        ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
        split_sentences = []
        split_points = [idx for idx, token in enumerate(ref_tokens) if token in (".", "Ġ.")]
        if split_points:
            for split_point in split_points:
                split_point += 1
                if split_sentences:
                    split_sentences.append(input_ids[prev_point : split_point])
                else:
                    split_sentences.append(input_ids[: split_point])
                prev_point = split_point
            split_sentences.append(input_ids[prev_point :])

            random.shuffle(split_sentences)
            
        input_ids = []
        for split_sentence in split_sentences:
            input_ids.extend(split_sentence)
        
        return input_ids