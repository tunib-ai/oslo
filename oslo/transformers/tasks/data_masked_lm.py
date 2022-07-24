import warnings
import logging
from typing import Dict, List, Optional, Any
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.transformers.tasks.data_utils import PARALLEL_KEY
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator

try:
    from transformers import DataCollatorForLanguageModeling
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logging.captureWarnings(True)


class ProcessorForMaskedLM(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int = 512) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        assert (
            self._tokenizer.eos_token_id is not None
            or self._tokenizer.sep_token_id is not None
        ), "The 'eos token' or 'sep token' must be defined."

        self._chunk_size = max_length - 2
        self.eos_text_id = (
            self._tokenizer.eos_token_id
            if self._tokenizer.eos_token_id is not None
            else self._tokenizer.sep_token_id
        )

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
            input_ids += [self.eos_text_id]
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


class DataCollatorForMaskedLM(DataCollatorForLanguageModeling):
    """
    Processing training examples to mini-batch for Roberta (mlm).
    """

    def __init__(
        self,
        processor: ProcessorForMaskedLM,
        mlm_probability: float = 0.15,
        parallel_context: Optional[ParallelContext] = None,
    ) -> None:
        if mlm_probability >= 1.0:
            warnings.warn("MLM Probability is greater than 1.0")

        if not isinstance(processor, ProcessorForMaskedLM):
            warnings.warn(
                "DataCollatorForRobertaPretraining is suitable for ProcessorForMaskedLM."
            )

        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.local_world_size = 1
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.local_world_size
            if self.local_world_size > 1
            else None,
        )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                parallel_key=PARALLEL_KEY["mlm"],
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
