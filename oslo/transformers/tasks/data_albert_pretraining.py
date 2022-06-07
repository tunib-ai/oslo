from typing import Any, Dict, List, Optional
import random
import torch
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.torch.distributed import ParallelContext, ParallelMode
try:
    from transformers import DataCollatorForLanguageModeling
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForAlbertPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
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
        pad_to_multiple_of: Optional[int] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.parallel_context = parallel_context
        if parallel_context is not None:
            self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
            self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
        

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_sop_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        
        if self.parallel_context is None:
            return batch
        else:
            for key, value in batch.items():
                if value.dim() < 2:
                    continue

                batch_size, seq_length = value.size()

                if seq_length % self.local_world_size != 0:
                    required_length = ((seq_length // self.local_world_size) + 1) * self.local_world_size
                    difference = required_length - seq_length

                    if key == "labels":
                        pads = torch.full([batch_size, difference], fill_value=-100, dtype=value.dtype)
                    elif key == "token_type_ids":
                        pads = torch.full([batch_size, difference], fill_value=1, dtype=value.dtype)
                    elif key == "attention_mask":
                        pads = torch.full([batch_size, difference], fill_value=0, dtype=value.dtype)
                    else:
                        pads = torch.full([batch_size, difference], fill_value=self.tokenizer.pad_token_id, dtype=value.dtype)
                    
                    value = torch.cat([value, pads], axis=1)
                
                value = value.chunk(
                    self.local_world_size,
                    dim=1,
                )[self.local_rank]

                if not value.is_contiguous():
                    value = value.contiguous()
                
                batch[key] = value

            return batch


    def _prepare_sop_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_a, token_b)
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_a, token_b)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
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