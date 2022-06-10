import warnings
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets.arrow_dataset import Batch
from oslo.transformers.tasks.data_base import BaseProcessor
from oslo.torch.distributed import ParallelContext, ParallelMode

try:
    from transformers import (
        T5Tokenizer,
        T5TokenizerFast,
    )
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


class ProcessorForT5Pretraining(BaseProcessor):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        mean_noise_span_length: float = 3.0,
    ) -> None:
        super().__init__(model_name_or_path, max_length)

        if not isinstance(self._tokenizer, (T5Tokenizer, T5TokenizerFast)):
            warnings.warn(
                "PorcessorForT5Pretraining is only suitable for T5Tokenizer-like tokenizers."
            )

        (
            self._chunk_size,
            self.target_chunk_size,
        ) = self.compute_input_and_target_lengths(
            max_length, mlm_probability, mean_noise_span_length
        )
        self._max_length = max_length
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length

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
            self._buffer.extend(input_ids)

            while len(self._buffer) >= self._chunk_size:
                chunk_ids = self._buffer[: self._chunk_size]
                dict_of_training_examples["input_ids"].append(chunk_ids)
                self._buffer = self._buffer[self._chunk_size :]

        return dict_of_training_examples

    def compute_input_and_target_lengths(
        self, inputs_length, noise_density, mean_noise_span_length
    ):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
        Training parameters to avoid padding with random_spans_noise_mask.
        When training a model with random_spans_noise_mask, we would like to set the other
        training hyperparmeters in a way that avoids padding.
        This function helps us compute these hyperparameters.
        We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
        and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
        This function tells us the required number of tokens in the raw example (for split_tokens())
        as well as the length of the encoded targets. Note that this function assumes
        the inputs and targets will have EOS appended and includes that in the reported length.
        Args:
            inputs_length: an integer - desired length of the tokenized inputs sequence
            noise_density: a float
            mean_noise_span_length: a float
        Returns:
            tokens_length: length of original text in tokens
            targets_length: an integer - length in tokens of encoded targets sequence
        """

        def _tokens_length_to_inputs_length_targets_length(tokens_length):
            num_noise_tokens = int(round(tokens_length * noise_density))
            num_nonnoise_tokens = tokens_length - num_noise_tokens
            num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
            # inputs contain all nonnoise tokens, sentinels for all noise spans
            # and one EOS token.
            _input_length = num_nonnoise_tokens + num_noise_spans + 1
            _output_length = num_noise_tokens + num_noise_spans + 1
            return _input_length, _output_length

        tokens_length = inputs_length

        while (
            _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
            <= inputs_length
        ):
            tokens_length += 1

        inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
            tokens_length
        )

        # minor hack to get the targets length to be equal to inputs length
        # which is more likely to have been set to a nice round number.
        if noise_density == 0.5 and targets_length > inputs_length:
            tokens_length -= 1
            targets_length -= 1
        return tokens_length, targets_length


class DataCollatorForT5Pretraining:
    """
    Processing training examples to mini-batch for T5 baseline pretraining (replace spans).
    """

    def __init__(
        self,
        processor: ProcessorForT5Pretraining,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.tokenizer = processor._tokenizer
        self.noise_density = processor.mlm_probability
        self.mean_noise_span_length = processor.mean_noise_span_length
        self.input_length = processor._max_length
        self.target_length = processor.target_chunk_size
        self.parallel_context = parallel_context
        if parallel_context is not None:
            self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
            self.local_world_size = parallel_context.get_world_size(
                ParallelMode.SEQUENCE
            )
            self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.tensor]:

        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        if self.parallel_context is None:
            batch = {key: torch.from_numpy(value) for key, value in batch.items()}
        else:
            for key, value in batch.items():
                value = torch.from_numpy(value)
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
                    else:
                        pads = torch.full(
                            [batch_size, difference],
                            fill_value=self.pad_token_id,
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

            batch["attention_mask"] = (
                (batch["input_ids"] != self.pad_token_id)
                .clone()
                .detach()
                .to(torch.int64)
            )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
