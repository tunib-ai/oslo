import torch
from typing import Dict, List, Optional, Union, Any
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.torch.distributed import ParallelContext, ParallelMode

try:
    from transformers import DataCollatorForLanguageModeling
    from transformers.tokenization_utils import BatchEncoding
    from transformers.data.data_collator import _torch_collate_batch
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForRobertaPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {}

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            add_special_tokens=False,
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
                    add_special_tokens=True,
                )

                training_example[
                    "special_tokens_mask"
                ] = self._tokenizer.get_special_tokens_mask(
                    training_example["input_ids"], already_has_special_tokens=True
                )

                for key in training_example.keys():
                    if key not in dict_of_training_examples:
                        dict_of_training_examples.setdefault(key, [])
                    dict_of_training_examples[key].append(training_example[key])

                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples


class DataCollatorForRobertaPretraining(DataCollatorForLanguageModeling):
    """
    Processing training examples to mini-batch for Roberta (mlm).
    """

    def __init__(
        self,
        processor: ProcessorForRobertaPretraining,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        parallel_context: Optional[ParallelContext] = None,
    ) -> None:
        super().__init__(
            tokenizer=processor._tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.parallel_context = parallel_context
        if parallel_context is not None:
            self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
            self.local_world_size = parallel_context.get_world_size(
                ParallelMode.SEQUENCE
            )

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        if self.parallel_context is None:
            return batch
        else:
            for key, value in batch.items():
                batch_size, seq_length = value.size()

                if seq_length % self.local_world_size != 0:
                    required_length = (
                        (seq_length // self.local_world_size) + 1
                    ) * self.local_world_size
                    difference = required_length - seq_length

                    if key == "labels":
                        pads = torch.full(
                            [batch_size, difference], fill_value=-100, dtype=value.dtype
                        )
                    elif key == "attention_mask":
                        pads = torch.full(
                            [batch_size, difference], fill_value=0, dtype=value.dtype
                        )
                    else:
                        pads = torch.full(
                            [batch_size, difference],
                            fill_value=self.tokenizer.pad_token_id,
                            dtype=value.dtype,
                        )

                    value = torch.cat([value, pads], axis=1)

                value = value.chunk(
                    self.local_world_size,
                    dim=1,
                )[self.local_rank]

                if not value.is_contiguous():
                    value = value.contiguous()

                batch[key] = value

            return batch
