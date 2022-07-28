from typing import Any, Dict, List, Optional
import random
import warnings
import logging
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor, ParallelKeys
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator

try:
    from transformers import DataCollatorForLanguageModeling
    from transformers import (
        AlbertTokenizer,
        AlbertTokenizerFast,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logging.captureWarnings(True)


class ProcessorForAlbertPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

        if not isinstance(self._tokenizer, (AlbertTokenizer, AlbertTokenizerFast)):
            warnings.warn(
                "ProcessorForAlbertPretraining is only suitable for AlbertTokenizer-like tokenizers."
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
            input_ids += [self._tokenizer.eos_token_id]
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class DataCollatorForAlbertPretraining(DataCollatorForLanguageModeling):
    """
    Processing training examples to mini-batch for Albert (mlm+sop).
    """

    def __init__(
        self,
        processor: ProcessorForAlbertPretraining,
        mlm_probability: float = 0.15,
        label_pad_token_id: int = -100,
        parallel_context: Optional[ParallelContext] = None,
    ):
        if mlm_probability >= 1.0:
            warnings.warn("MLM Probability is greater than 1.0")

        assert isinstance(
            processor, ProcessorForAlbertPretraining
        ), "DataCollatorForAlbertPretraining is only suitable for ProcessorForAlbertPretraining."

        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_type_id = self.tokenizer.pad_token_type_id
        self.label_pad_token_id = label_pad_token_id
        self.local_world_size = 1
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_sop_from_examples(examples)
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.local_world_size
            if self.local_world_size > 1
            else None,
        )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        if self.label_pad_token_id != -100:
            batch["labels"].masked_fill_(
                batch["labels"] == -100, self.label_pad_token_id
            )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                parallel_keys=ParallelKeys.ALBERT,
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def _prepare_sop_from_examples(
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
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids, already_has_special_tokens=True
            )
            sentence_order_label = 1 if reverse else 0

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "special_tokens_mask": special_tokens_mask,
                    "sentence_order_label": sentence_order_label,
                }
            )
        return output_examples

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
