from typing import Any, Dict, List, Optional, Union
import logging
import warnings
import torch
import numpy as np
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor, PARALLEL_KEY
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator

try:
    from transformers import DataCollatorForSeq2Seq
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForSummarization(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "labels" in column_names
        ), "The name of dataset column that you want to use as a summary must be 'labels'"

        assert (
            "text" in column_names
        ), "The name of dataset column that you want to use as a text must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {}

        dict_of_training_examples["input_ids"] = self._tokenizer(
            examples["text"],
            truncation=True,
            max_length=self._max_length,
            verbose=False,
        )["input_ids"]

        dict_of_training_examples["labels"] = self._tokenizer(
            examples["labels"],
            truncation=True,
            max_length=self._max_length,
            verbose=False,
        )["input_ids"]

        return dict_of_training_examples


class DataCollatorForSummarization(DataCollatorForSeq2Seq):
    """
    Processing training examples to mini-batch (summarization).
    """

    def __init__(
        self,
        processor: ProcessorForSummarization,
        pad_to_multiple_of: Optional[int] = None,
        parallel_context: Optional[ParallelContext] = None,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = "longest",
        label_pad_token_id: int = -100,
    ):
        if not isinstance(processor, ProcessorForSummarization):
            warnings.warn(
                "DataCollatorForSummarization is suitable for ProcessorForSummarization."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer
        self.model = model
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.local_world_size = 0
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [feature["labels"] for feature in features]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels)
        self.pad_to_multiple_of = (
            self.local_world_size
            if self.local_world_size > 1
            else self.pad_to_multiple_of
        )
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of

        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (
                max_label_length - len(feature["labels"])
            )
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder
                    if padding_side == "right"
                    else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate(
                    [feature["labels"], remainder]
                ).astype(np.int64)
            else:
                feature["labels"] = np.concatenate(
                    [remainder, feature["labels"]]
                ).astype(np.int64)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch["labels"]
            )
            batch["decoder_input_ids"] = decoder_input_ids
            batch["decoder_attention_mask"] = torch.where(
                decoder_input_ids == self.tokenizer.pad_token_id,
                0,
                torch.ones_like(decoder_input_ids),
            )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                tokenizer=self.tokenizer,
                parallel_key=PARALLEL_KEY["summarization"],
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
