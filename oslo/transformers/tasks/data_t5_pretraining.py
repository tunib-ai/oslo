import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
try:
    from transformers import (
        T5Tokenizer,
        T5TokenizerFast,
    )
    from transformers.data.data_collator import _torch_collate_batch
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForT5PrefixLanguageModeling(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "labels": []
        }

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )['input_ids']

        list_of_input_ids, list_of_labels = self._prepare_for_prefix_language_modeling(list_of_input_ids)
        for input_ids, labels in zip(list_of_input_ids, list_of_labels):
            if len(input_ids) <= self._max_length and len(labels) <= self._max_length:
                dict_of_training_examples["input_ids"].append(input_ids)
                dict_of_training_examples["labels"].append(labels)

        return dict_of_training_examples
    
    def _prepare_for_prefix_language_modeling(self, list_of_input_ids: List[List[int]]) -> Tuple[List[List[str]], List[List[str]]]:
        inputs, targets = [], []
        for input_ids in list_of_input_ids:
            seq_length = len(input_ids)
            if seq_length >= 3:
                start, end = seq_length // 3, seq_length // 3 * 2
                split_position = random.randrange(start, end)
                inputs.append(input_ids[:split_position])
                targets.append(input_ids[split_position:])
        
        return inputs, targets


class DataCollatorForT5PrefixLanguageModeling:
    """
    Processing training examples to mini-batch for T5 (prefix language modeling).
    """

    def __init__(
        self,
        processor: ProcessorForT5PrefixLanguageModeling,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]

        batch = {
            "input_ids": _torch_collate_batch(
                input_ids,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
            "labels": _torch_collate_batch(
                labels,
                tokenizer=self.tokenizer,
                pad_to_multiple_of=self.pad_to_multiple_of,
            ),
        }
        return batch


class ProcessorForBERTstylePretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
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


class DataCollatorForBERTstylePretraining:
    """
    Processing training examples to mini-batch for T5 BERT_style masking language modeling (mlm+wwm).
    """

    def __init__(
        self,
        processor: ProcessorForBERTstylePretraining,
        mlm_probability: float,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_token = processor._tokenizer.additional_special_tokens[0]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_wwm_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch_mask = batch.pop("mask_label")
        batch["input_ids"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
        return batch

    def _prepare_wwm_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)
            ref_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            mask_label = self._whole_word_mask(ref_tokens)

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "mask_label": mask_label,
                }
            )
        return output_examples
    
    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 90% MASK, 10% random. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 90% of the time, we replace masked input tokens with mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.9)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (T5Tokenizer, T5TokenizerFast)):
            warnings.warn(
                "DataCollatorForBartTokenMasking is only suitable for T5Tokenizer-like tokenizers. "
            )

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "</s>":
                continue
            
            if not cand_indexes:
                cand_indexes.append([i])
            elif token.startswith("▁"):
                cand_indexes.append([i])
            else:
                cand_indexes[-1].append(i)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels