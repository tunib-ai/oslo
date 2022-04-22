from typing import Any, Dict, List, Optional, Union

from transformers import GPT2ForTokenClassification, AutoTokenizer
from transformers.file_utils import PaddingStrategy
from datasets import Dataset, DatasetDict
from data_utils import BaseProcessor


class ProcessorForTokenClassification(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: Optional[int] = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
        self._max_length = max_length
        self._chunk_size = max_length
        self._buffer = []
    
    def __call__(self, examples: Union[DatasetDict, Dataset]) -> Dict[str, List[int]]:
        dict_of_training_examples: Dict[str, List[int]] = self._tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            verbose=False,
        )
        all_labels = examples["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = dict_of_training_examples.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
        
        dict_of_training_examples["labels"] = new_labels
        return dict_of_training_examples
    
    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        
        return new_labels


class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """

    def __init__(
        self,
        tokenizer: ProcessorForTokenClassification,
        padding: Union[bool, str, PaddingStrategy] = "longest",
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding
        if self.tokenizer._pad_token is None:
            self.tokenizer._pad_token = self.tokenizer._eos_token

    def __call__(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch