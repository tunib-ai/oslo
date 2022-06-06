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
        max_length, 
        dataset,
        batch_size=1024,
        pad_to_multiple_of=None,
        parallel_context=None,
    ):
        self.processor = ProcessorForCausalLM(model_name, max_length)
        self.data_collator = DataCollatorForCausalLM(
            self.processor, pad_to_multiple_of=pad_to_multiple_of
        )
        self.sp_data_collator = DataCollatorForCausalLM(
            self.processor, pad_to_multiple_of=pad_to_multiple_of, parallel_context=parallel_context
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.pad_to_multilple_of = pad_to_multiple_of
        self.parallel_context = parallel_context

    def __call__(self):
        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {self.max_length}",
            f"Batch size: {self.batch_size}",
            f"Pad to multiple of: {self.pad_to_multilple_of}\n",
            sep="\n"
        )
        self.processed_dataset = self.dataset.map(
            self.processor,
            batched=True,
            remove_columns = self.dataset['train'].column_names
        )
        self.processed_dataset.cleanup_cache_files()

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer = self.data_collator.tokenizer
            print("pad_token is set.")

        dataloader = DataLoader(
            self.processed_dataset['train'], self.batch_size, shuffle=True, collate_fn=self.data_collator
        )
        
        batch = next(iter(dataloader))
        self._batch_check(batch, num_samples=2, check_token=True)

        self._length_check(
            dataloader, 
            "input_ids", 
            self.max_length, 
            self.pad_to_multilple_of, 
            must_equal_to_max_length=True
        )
        self._length_check(
            dataloader, 
            "labels", 
            self.max_length, 
            self.pad_to_multilple_of, 
            must_equal_to_max_length=True
        )

        if self.parallel_context is not None:
            self._test_sp_collator(self.processed_dataset, self.batch_size)
        
        print("---------- Test Pass ----------\n")


if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    test_1 = TestDataCausalLM("gpt2", 256, dataset, 1024, 3)
    test_1()

    test_2 = TestDataCausalLM("gpt2", 512, dataset, 1024)
    test_2()

    test_3 = TestDataCausalLM("facebook/bart-base", 64, dataset, 32)
    test_3()

    test_4 = TestDataCausalLM("t5-small", 128, dataset, 1024)
    test_4()

    # parallel_context = ParallelContext.from_torch(sequence_parallel_size=3)
    # test_sp = TestDataCausalLM("gpt2", 256, dataset, 1024, 3, parallel_context)
    # test_sp()