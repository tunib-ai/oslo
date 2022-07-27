from typing import Any, Dict, List, Optional
import logging
import warnings
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor, ParallelKey
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.utils.data.data_collators import SequenceDataParallelCollator

try:
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


class ProcessorForSequenceClassification(BaseProcessor):
    def __init__(
        self, model_name_or_path: str, max_length: int, is_text_pair: bool = False
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self.is_text_pair = is_text_pair

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "labels" in column_names
        ), "The name of dataset column that you want to use as a label must be 'labels'"

        if self.is_text_pair:
            assert (
                "text1" in column_names and "text2" in column_names
            ), "The name of dataset columns that you want to tokenize must be 'text1' and 'text2'"

            dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
                examples["text1"],
                examples["text2"],
                truncation=True,
                max_length=self._max_length,
                verbose=False,
            )
        else:
            assert (
                "text" in column_names
            ), "The name of dataset column that you want to tokenize must be 'text'"

            dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=self._max_length,
                verbose=False,
            )

        dict_of_training_examples["labels"] = examples["labels"]

        return dict_of_training_examples


class DataCollatorForSequenceClassification:
    """
    Processing training examples to mini-batch for Gpt2 (sequence classification).
    """

    def __init__(
        self,
        processor: ProcessorForSequenceClassification,
        padding: PaddingStrategy = "longest",
        parallel_context: Optional[ParallelContext] = None,
    ):
        if not isinstance(processor, ProcessorForSequenceClassification):
            warnings.warn(
                "DataCollatorForSequenceClassification is suitable for ProcessorForSequenceClassification."
            )

        if processor._tokenizer.pad_token is None:
            warnings.warn(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )

        self.tokenizer = processor._tokenizer
        self.padding = padding
        self.local_world_size = 1
        if parallel_context is not None:
            self.set_parallel_context(parallel_context)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
            pad_to_multiple_of=self.local_world_size
            if self.local_world_size > 1
            else None,
        )

        if self.local_world_size > 1:
            sp_collate_fn = SequenceDataParallelCollator(
                parallel_key=ParallelKey.SEQ_CLS,
                parallel_context=self.parallel_context,
            )
            return sp_collate_fn(**batch)

        return batch

    def set_parallel_context(self, parallel_context: ParallelContext):
        self.parallel_context = parallel_context
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
