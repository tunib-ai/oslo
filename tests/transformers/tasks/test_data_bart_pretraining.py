import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)
from tests.transformers.tasks.test_data_base import TestDataBinarization
try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataBartPretraining(TestDataBinarization):
    def __init__(
        self,
        model_name,
        parallel_context = None,
    ):
        self.processor = ProcessorForBartPretraining(model_name)
        self.data_collator = DataCollatorForBartPretraining(self.processor)
        self.sp_data_collator = DataCollatorForBartPretraining(
            self.processor, parallel_context=parallel_context
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.parallel_context = parallel_context

    def __call__(
        self,
        max_length,
        dataset,
        mlm_probability=0.15,
        possion_lambda=3,
        batch_size = 1024,
        pad_to_multiple_of = None,
        batch_check_num_sample = 2,
        batch_check_tokens = False,
    ):
        self.processor._chunk_size = max_length - 1
        self.processor._max_length = max_length
        self.data_collator.pad_to_multiple_of = pad_to_multiple_of
        self.data_collator.mlm_probability = mlm_probability
        self.data_collator.possion_lambda = possion_lambda

        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {max_length}",
            f"Batch size: {batch_size}",
            f"MLM probability: {mlm_probability}",
            f"Possion Lambda: {possion_lambda}",
            f"Pad to multiple of: {pad_to_multiple_of}\n",
            sep="\n"
        )
        processed_dataset = dataset.map(
            self.processor,
            batched=True,
            remove_columns = dataset['train'].column_names
        )
        processed_dataset.cleanup_cache_files()

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer = self.data_collator.tokenizer
            print("pad_token is set.")

        dataloader = DataLoader(
            processed_dataset['train'], batch_size, shuffle=True, collate_fn=self.data_collator
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
            must_be_equal_to_max_length=False
        )

        self._length_check(
            dataloader, 
            "labels", 
            max_length, 
            pad_to_multiple_of, 
            must_be_equal_to_max_length=True
        )

        self.mask_ratio_check(dataloader)

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)
        
        print("---------- Test Pass ----------\n")
    
    def mask_ratio_check(self, dataloader):
        mask_token_id = self.tokenizer.mask_token_id
        pad_token_id = self.tokenizer.pad_token_id

        for batch in dataloader:
            batch_size, seq_length = batch['labels'].size()

            # Verify that the mask token ratio is aligned to a predetermined percentage
            num_pad_tokens = torch.sum(batch['labels'] == pad_token_id)
            num_tokens = batch_size * (seq_length - 1) - num_pad_tokens
            mean_num_mask_tokens = torch.sum(batch['input_ids'] == mask_token_id) * self.data_collator.possion_lambda
            mlm_probability = mean_num_mask_tokens / num_tokens
            assert(
                torch.isclose(mlm_probability, torch.tensor(self.data_collator.mlm_probability), atol=0.005)
            ), f"Mask ratio({mlm_probability:.6f}) is different from the predefined one({self.data_collator.mlm_probability})"

        print("---- mask ratio test pass ----\n")


if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    bart_test = TestDataBartPretraining("facebook/bart-base")
    # bart_test(1024, dataset)
    bart_test(512, dataset, pad_to_multiple_of=3)
    bart_test(512, dataset, 0.2)
    bart_test(1024, dataset, 0.2, 4)
    bart_test(512, dataset, 0.2, 2)
    bart_test(512, dataset, 0.3)
    # bart_test(256, dataset, 0.3)
    # bart_test(256, dataset, 0.3, 2)