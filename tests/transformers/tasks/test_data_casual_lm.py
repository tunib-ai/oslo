import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
    DataCollatorForCausalLM,
)
from tests.transformers.tasks.test_data_base import TestDataBinarization

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataCausalLM(TestDataBinarization):
    def __init__(
        self,
        model_name,
        parallel_context=None,
    ):
        self.processor = ProcessorForCausalLM(model_name)
        self.data_collator = DataCollatorForCausalLM(self.processor)
        self.sp_data_collator = DataCollatorForCausalLM(
            self.processor, parallel_context=parallel_context
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.parallel_context = parallel_context

    def __call__(
        self,
        max_length,
        dataset,
        batch_size=1024,
        pad_to_multiple_of=None,
        batch_check_num_sample=2,
        batch_check_tokens=False,
        must_be_equal_to_max_length=True,
    ):

        self.processor._chunk_size = max_length
        self.data_collator.pad_to_multiple_of = pad_to_multiple_of

        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {max_length}",
            f"Batch size: {batch_size}",
            f"Pad to multiple of: {pad_to_multiple_of}\n",
            sep="\n",
        )
        processed_dataset = dataset.map(
            self.processor, batched=True, remove_columns=dataset["train"].column_names
        )
        processed_dataset.cleanup_cache_files()

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer = self.data_collator.tokenizer
            print("pad_token is set.")

        dataloader = DataLoader(
            processed_dataset["train"],
            batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        batch = next(iter(dataloader))
        self._batch_check(
            batch, num_samples=batch_check_num_sample, check_token=batch_check_tokens
        )

        self._length_check(
            dataloader,
            "input_ids",
            max_length,
            pad_to_multiple_of,
            must_be_equal_to_max_length=must_be_equal_to_max_length,
        )
        self._length_check(
            dataloader,
            "labels",
            max_length,
            pad_to_multiple_of,
            must_be_equal_to_max_length=must_be_equal_to_max_length,
        )

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)

        print("---------- Test Pass ----------\n")


if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    gpt2_test = TestDataCausalLM("gpt2")
    gpt2_test(512, dataset, 1024)
    gpt2_test(256, dataset, 1024, 3)

    bart_test = TestDataCausalLM("facebook/bart-base")
    bart_test(64, dataset, 32)

    t5_test = TestDataCausalLM("t5-small")
    t5_test(128, dataset, 1024)

    # parallel_context = ParallelContext.from_torch(sequence_parallel_size=3)
    # gpt2_sp_test = TestDataCausalLM("gpt2", parallel_context)
    # gpt2_sp_test(256, dataset, 1024)
