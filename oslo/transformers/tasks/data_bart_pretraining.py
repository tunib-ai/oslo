import random
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import warnings
import logging
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor, PARALLEL_KEY
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator
from oslo.transformers.models.bart.modeling_bart import shift_tokens_right

try:
    from transformers import (
        BartTokenizer,
        BartTokenizerFast,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logging.captureWarnings(True)


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
        permute_sentence: bool = True,
        parallel_context: Optional[ParallelContext] = None,
    ):
        if mlm_probability >= 1.0:
            warnings.warn("MLM Probability is greater than 1.0")

        assert isinstance(
            processor, ProcessorForBartPretraining
        ), "DataCollatorForBartPretraining is only suitable for ProcessorForBartPretraining."

        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.possion_lambda = possion_lambda
        self.permute_sentence = permute_sentence
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.eos_token_id
        self.mask_token_id = processor._tokenizer.mask_token_id
        self.local_world_size = 0
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)
        self.get_start_indices = {
            max_idx: np.random.choice(max_idx, size=(max_idx,), replace=False)
            for max_idx in range(processor._chunk_size, 0, -1)
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_noise_text_from_examples(examples)

        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.local_world_size
            if self.local_world_size > 1
            else None,
        )

        if self.local_world_size > 1:
            batch_size, label_seq_length = batch["labels"].size()
            if label_seq_length % self.local_world_size != 0:
                label_required_length = (
                    (label_seq_length // self.local_world_size) + 1
                ) * self.local_world_size

                difference = label_required_length - label_seq_length
                label_pads = torch.full(
                    (batch_size, difference),
                    fill_value=-100,
                    dtype=batch["labels"].dtype,
                )
                batch["labels"] = torch.cat([batch["labels"], label_pads], axis=-1)

        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            batch["decoder_input_ids"] == self.pad_token_id,
            0,
            torch.ones_like(batch["decoder_input_ids"]),
        )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                tokenizer=self.tokenizer,
                parallel_key=PARALLEL_KEY["bart"],
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def _prepare_noise_text_from_examples(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            labels = chunk_ids[:]

            chunk_ids = self._text_infilling(chunk_ids)
            if self.permute_sentence:
                chunk_ids = self._sentence_permutation(chunk_ids)

            chunk_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)
            labels = self.tokenizer.build_inputs_with_special_tokens(labels)[1:]

            output_examples.append(
                {
                    "input_ids": chunk_ids,
                    "labels": labels,
                }
            )

        return output_examples

    def _text_infilling(self, input_ids: list):
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
            segment_lengths.sort(reverse=True)
            return segment_lengths

        temp_ids = input_ids
        while len(temp_ids) >= length:
            temp_ids = input_ids[:]
            noise_span_lengths = _possion_segmentation(num_noise_tokens)

            for noise_span_length in noise_span_lengths:
                max_idx = len(temp_ids) - noise_span_length + 1
                # get start index of mask span
                start_indices = self.get_start_indices[max_idx]
                for start_idx in start_indices:
                    if (
                        self.mask_token_id
                        in temp_ids[start_idx : start_idx + noise_span_length]
                    ):
                        continue
                    else:
                        temp_ids = (
                            temp_ids[:start_idx]
                            + [self.mask_token_id]
                            + temp_ids[start_idx + noise_span_length :]
                        )
                        # rotate start indices
                        self.get_start_indices[max_idx] = np.roll(start_indices, 1)
                        break

        input_ids = temp_ids

        return input_ids

    def _sentence_permutation(self, input_ids):
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

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
