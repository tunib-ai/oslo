import random
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import warnings
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
try:
    from transformers import (
        DataCollatorForWholeWordMask,
        BartTokenizer,
        BartTokenizerFast,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForBartPretraining(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        examples["text"] = self._split_by_full_stops(examples["text"])

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "mask_infilling_labels": []
        }

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )['input_ids']

        for input_ids in list_of_input_ids:
            self._buffer.extend(input_ids)

            mask_span_length = np.random.poisson(lam=3)
            while len(self._buffer) >= self._chunk_size + mask_span_length:
                chunk_ids = self._buffer[: self._chunk_size + mask_span_length]
                mask_chunk_ids = self._prepare_mask_infilling(chunk_ids, mask_span_length)

                dict_of_training_examples["input_ids"].append(mask_chunk_ids)
                dict_of_training_examples["mask_infilling_labels"].append(mask_span_length)

                self._buffer = self._buffer[self._chunk_size + mask_span_length :]

        return dict_of_training_examples
    
    def _split_by_full_stops(self, texts: List[str]) -> List[str]:
        splited_texts = []
        for text in texts:
            splited_sentences = ""
            for sentence in text.split("."):
                if sentence: # 빈 문장은 제거
                    splited_sentences += sentence + "." + self._tokenizer._sep_token
            
            if splited_sentences: # 빈 문서는 제거
                splited_texts.append(splited_sentences)
                
        return splited_texts
    
    def _prepare_mask_infilling(self, chunk_ids: List[int], mask_span_length: int) -> List[int]:
        seq_length = len(chunk_ids)
        start_position, end_position = 0, seq_length
        while self._tokenizer.sep_token_id in chunk_ids[start_position : end_position]:
            end = seq_length - mask_span_length
            start_position = random.randrange(0, end)
            end_position = start_position + mask_span_length
        mask_chunk_ids = chunk_ids[: start_position] + [self._tokenizer.mask_token_id] + chunk_ids[end_position+1 :]
        return mask_chunk_ids


class DataCollatorForBartPretraining:
    """
    Processing training examples to mini-batch for Bart (mask_infilling, sentence_permutation).
    """

    def __init__(
        self,
        processor: ProcessorForBartPretraining,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_sentence_permutaion_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        return batch

    def _prepare_sentence_permutaion_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output_examples = []
        for example in examples:
            chunk_ids = example["input_ids"]

            split_sentences = []
            split_points = filter(lambda x: chunk_ids[x] == self.tokenizer.sep_token_id, range(len(chunk_ids)))
            for split_point in split_points:
                split_point += 1
                if split_sentences:
                    split_sentences.append(chunk_ids[prev_point : split_point])
                else:
                    split_sentences.append(chunk_ids[: split_point])  
                prev_point = split_point
            split_sentences.append(chunk_ids[prev_point :])

            random.shuffle(split_sentences)
            input_ids = []
            for split_sentence in split_sentences:
                input_ids.extend(split_sentence)
            
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            chunk_ids = self.tokenizer.build_inputs_with_special_tokens(chunk_ids)

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "sentence_permutation_labels": chunk_ids,
                    "mask_infilling_labels": example["mask_infilling_labels"],
                }
            )
            
        return output_examples


class ProcessorForBartTokenMasking(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

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


class DataCollatorForBartTokenMasking(DataCollatorForWholeWordMask):
    """
    Processing training examples to mini-batch for Bart Token Masking (mlm+wwm).
    """

    def __init__(
        self,
        processor: ProcessorForBartTokenMasking,
        mlm_probability: float,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._prepare_wwm_from_examples(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch_mask = batch.pop("mask_label")
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
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
                    "mask_label": mask_label,
                }
            )
        return output_examples
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=1024):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BartTokenizer, BartTokenizerFast)):
            warnings.warn(
                "DataCollatorForBartTokenMasking is only suitable for BertTokenizer-like tokenizers. "
            )

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "<s>" or token == "</s>":
                continue
            
            if not cand_indexes:
                cand_indexes.append([i])
            elif token.startswith("Ġ"):
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


class ProcessorForBartTokenDeletion(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2
        self.special_tokens = self._tokenizer.special_tokens_map.values()

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "labels": [],
        }

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

            while len(self._buffer) >= self._chunk_size+1:
                chunk_ids = self._buffer[: self._chunk_size+1]
                chunk_ids, mask_index = self._prepare_mask_deletion(chunk_ids)
                dict_of_training_examples["input_ids"].append(chunk_ids)
                dict_of_training_examples["labels"].append(mask_index)
                self._buffer = self._buffer[self._chunk_size+1 :]
        
        # for input_ids in list_of_input_ids:
        #     input_ids += [self._tokenizer.eos_token_id]
        #     self._buffer.extend(input_ids)

        #     indice_replaced = torch.bernoulli(torch.full((self._chunk_size,), 0.15)).bool()
        #     num_mask = sum(indice_replaced)
        #     while len(self._buffer) >= self._chunk_size + num_mask:
        #         chunk_ids = self._buffer[: self._chunk_size + num_mask]
        #         chunk_ids, masked_indice = self._prepare_mask_deletion_more_than_one_label(chunk_ids, indice_replaced)
        #         dict_of_training_examples["input_ids"].append(chunk_ids)
        #         dict_of_training_examples["labels"].append(masked_indice)
        #         self._buffer = self._buffer[self._chunk_size + num_mask :]

        return dict_of_training_examples
    
    def _prepare_mask_deletion(self, chunk_ids):
        seq_length = len(chunk_ids)
        mask_index = random.randrange(0, seq_length)
        masked_id = chunk_ids[mask_index]
        masked_token = self._tokenizer.convert_ids_to_tokens(masked_id)
        
        while masked_token in self.special_tokens:
            mask_index = random.randrange(0, seq_length)
            masked_id = chunk_ids[mask_index]
            masked_token = self._tokenizer.convert_ids_to_tokens(masked_id)
        
        chunk_ids.pop(mask_index)

        return chunk_ids, mask_index
    
    # def _prepare_mask_deletion_more_than_one_label(self, chunk_ids, indice_replaced):
    #     masked_indice = filter(lambda x: indice_replaced[x], range(len(chunk_ids)))

    #     masked_indice = list(masked_indice)[::-1]
    #     for masked_index in masked_indice:
    #         chunk_ids.pop(masked_index)
    #     masked_indice = masked_indice[::-1]

    #     return chunk_ids, masked_indice


class DataCollatorForBartTokenDeletion:
    """
    Processing training examples to mini-batch for Bart (mask_deletion).
    """

    def __init__(
        self,
        processor: ProcessorForBartTokenDeletion,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._build_inputs_with_special_tokens(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        return batch
    
    def _build_inputs_with_special_tokens(self, examples):
        output_examples = []
        for example in examples:
            input_ids = example['input_ids']
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            label = example['labels'] + 1

            # label = [label+1 for label in example['labels']]

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "labels": label,
                }
            )
        return output_examples


class ProcessorForBartDocumentRotation(BaseProcessor):
    def __init__(self, model_name_or_path: str, max_length: int) -> None:
        super().__init__(model_name_or_path=model_name_or_path, max_length=max_length)
        self._chunk_size = max_length - 2

    def __call__(self, examples: Batch) -> Dict[str, List[int]]:
        column_names = [k for k, v in examples.items()]
        assert (
            "text" in column_names
        ), "The name of dataset column that you want to tokenize must be 'text'"

        dict_of_training_examples: Dict[str, List[int]] = {
            "input_ids": [],
            "labels": [],
        }

        list_of_input_ids: List[List[int]] = self._tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            verbose=False,
        )['input_ids']

        for input_ids in list_of_input_ids:
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                rotated_ids, start_point = self._prepare_for_document_rotation(chunk_ids)
                dict_of_training_examples["input_ids"].append(rotated_ids)
                dict_of_training_examples["labels"].append(start_point)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples
    
    def _prepare_for_document_rotation(self, chunk_ids):
        seq_length = len(chunk_ids)
        random_point = random.randrange(0, seq_length)
        rotated_ids = chunk_ids[random_point:] + chunk_ids[:random_point]
        start_point = len(chunk_ids[random_point:])

        return rotated_ids, start_point


class DataCollatorForBartDocumentRotation:
    """
    Processing training examples to mini-batch for Bart (document rotation).
    """

    def __init__(
        self,
        processor: ProcessorForBartDocumentRotation,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = self._build_inputs_with_special_tokens(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        return batch
    
    def _build_inputs_with_special_tokens(self, examples):
        output_examples = []
        for example in examples:
            input_ids = example['input_ids']
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            label = example['labels'] + 1

            output_examples.append(
                {
                    "input_ids": input_ids,
                    "labels": label,
                }
            )
        return output_examples