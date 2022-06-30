import random
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import warnings
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.torch.distributed import ParallelContext, ParallelMode

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
        max_length: int = 1024,
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
            add_special_tokens=False,
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
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.possion_lambda = possion_lambda
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = processor._tokenizer.mask_token_id
        self.parallel_context = parallel_context
        if parallel_context is not None:
            self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
            self.local_world_size = parallel_context.get_world_size(
                ParallelMode.SEQUENCE
            )
            self.pad_to_multiple_of = self.local_world_size

        if not isinstance(processor, ProcessorForBartPretraining):
            warnings.warn(
                "DataCollatorForBartPretraining is only suitable for ProcessorForBartPretraining."
            )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_noise_text_from_examples(examples)

        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if self.pad_to_multiple_of is not None:
            batch_size, label_seq_length = batch["labels"].size()
            if label_seq_length % self.pad_to_multiple_of != 0:
                label_required_length = (
                    (label_seq_length // self.pad_to_multiple_of) + 1
                ) * self.pad_to_multiple_of

                difference = label_required_length - label_seq_length
                label_pads = torch.full(
                    (batch_size, difference),
                    fill_value=self.pad_token_id,
                    dtype=batch["labels"].dtype,
                )
                batch["labels"] = torch.cat([batch["labels"], label_pads], axis=1)

        if self.parallel_context is not None:
            for key, value in batch.items():
                value = value.chunk(
                    self.local_world_size,
                    dim=1,
                )[self.local_rank]

                if not value.is_contiguous():
                    value = value.contiguous()

                batch[key] = value

        return batch

    def _prepare_noise_text_from_examples(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            labels = chunk_ids[:]

            chunk_ids = self.text_infilling(chunk_ids)
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

    def text_infilling(self, input_ids: list):
        length = len(input_ids)
        num_noise_tokens = int(np.round(length * self.mlm_probability))

        # pick the lengths of the noise spans
        def _possion_segmentation(num_noise_tokens):
            segment_lengths = []
            while sum(segment_lengths) < num_noise_tokens:
                span_length = np.random.poisson(lam=self.possion_lambda)
                segment_lengths.append(span_length)

            difference = sum(segment_lengths) - num_noise_tokens
            segment_lengths[-1] = segment_lengths[-1] - difference
            return segment_lengths

        temp_ids = input_ids
        while len(temp_ids) >= length:
            temp_ids = input_ids[:]
            noise_span_lengths = _possion_segmentation(num_noise_tokens)

            for noise_span_length in noise_span_lengths:
                max_idx = len(temp_ids) - noise_span_length + 1
                start_idx = np.random.choice(max_idx)
                while (
                    self.mask_token_id
                    in temp_ids[start_idx : start_idx + noise_span_length]
                ):
                    start_idx = np.random.choice(max_idx)
                temp_ids = (
                    temp_ids[:start_idx]
                    + [self.mask_token_id]
                    + temp_ids[start_idx + noise_span_length :]
                )

        input_ids = temp_ids

        return input_ids

    def sentence_permutation(self, input_ids):
        ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        split_sentences = []
        split_points = [
            idx for idx, token in enumerate(ref_tokens) if token in (".", "Ġ.")
        ]

        if split_points:
            prev_point = 0
            for split_point in split_points:
                split_point += 1
                split_sentences.append(input_ids[prev_point:split_point])
                prev_point = split_point
            split_sentences.append(input_ids[prev_point:])

            random.shuffle(split_sentences)

            input_ids = []
            for split_sentence in split_sentences:
                input_ids.extend(split_sentence)

        return input_ids
