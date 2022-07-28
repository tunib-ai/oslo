from typing import Any, Dict, List, Optional
import random
import warnings
import logging
import torch
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor, ParallelKeys, pad_labels
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator

try:
    from transformers import DataCollatorForWholeWordMask
    from transformers import (
        BertTokenizer,
        BertTokenizerFast,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logging.captureWarnings(True)


class ProcessorForBertPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

        if not isinstance(self._tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "PorcessorForBertPretraining is suitable for BertTokenizer-like tokenizers."
            )

        self._chunk_size = max_length - 3

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


class DataCollatorForBertPretraining(DataCollatorForWholeWordMask):
    """
    Processing training examples to mini-batch for Bert (mlm+wwm+sop).
    """

    def __init__(
        self,
        processor: ProcessorForBertPretraining,
        mlm_probability: float = 0.15,
        label_pad_token_id: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        if mlm_probability >= 1.0:
            warnings.warn("MLM Probability is greater than 1.0")

        assert isinstance(
            processor, ProcessorForBertPretraining
        ), "DataCollatorForBertPretraining is only suitable for ProcessorForBertPretraining."

        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_type_id = self.tokenizer.pad_token_type_id
        self.label_pad_token_id = label_pad_token_id
        self.local_world_size = 1
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_wwm_and_sop_from_examples(examples)
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.local_world_size
            if self.local_world_size > 1
            else None,
        )

        batch_mask = batch.pop("mask_label")

        if self.local_world_size > 1:
            batch_mask = pad_labels(
                batch_mask,
                self.tokenizer,
                label_pad_token_id=0,
                pad_to_multiple_of=self.local_world_size,
            )

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], mask_labels=batch_mask
        )
        if self.label_pad_token_id != -100:
            batch["labels"].masked_fill_(
                batch["labels"] == -100, self.label_pad_token_id
            )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                parallel_keys=ParallelKeys.BERT,
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def _prepare_wwm_and_sop_from_examples(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            seq_length = len(chunk_ids)
            start, end = seq_length // 3, seq_length // 3 * 2
            split_position = random.randrange(start, end)
            reverse = random.random() < 0.5

            if reverse:
                token_a = chunk_ids[split_position:]
                token_b = chunk_ids[:split_position]
            else:
                token_a = chunk_ids[:split_position]
                token_b = chunk_ids[split_position:]

            input_ids = self.tokenizer.build_inputs_with_special_tokens(
                token_a, token_b
            )
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
                token_a, token_b
            )
            sentence_order_label = 1 if reverse else 0
            ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            mask_label = self._whole_word_mask(ref_tokens)

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "next_sentence_label": sentence_order_label,
                    "mask_label": mask_label,
                }
            )
        return output_examples

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
