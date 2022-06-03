from typing import Any, Dict, List, Optional
import logging
from datasets.arrow_dataset import Batch

from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.torch.distributed import ParallelContext, ParallelMode
try:
    from transformers.file_utils import PaddingStrategy
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


logger = logging.getLogger(__name__)


class ProcessorForSequenceClassification(BaseProcessor):
    def __init__(
        self, model_name_or_path: str, max_length: Optional[int] = None
    ) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"
        assert (
            "labels" in column_names
        ), "The name of dataset column that you want to use as a label must be 'labels'"

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
        tokenizer: ProcessorForSequenceClassification,
        pad_to_multiple_of: Optional[int] = None,
        padding: PaddingStrategy = "longest",
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.tokenizer = tokenizer._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.parallel_context = parallel_context
        if parallel_context is not None:
            self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
            self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.tokenizer.pad_token is None:
            logger.warning(
                "If pad token doesn't exist in tokenizer, it can be a problem when applying padding."
            )
        
        if self.parallel_context is None:
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        else:
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                pad_to_multiple_of=self.local_world_size,
                return_tensors="pt",
            )
            
            for key, value in batch.items():
                if value.dim() < 2:
                    continue
                
                value = value.chunk(
                    self.local_world_size,
                    dim=1,
                )[self.local_rank]

                if not value.is_contiguous():
                    value = value.contiguous()
                
                batch[key] = value

        return batch
