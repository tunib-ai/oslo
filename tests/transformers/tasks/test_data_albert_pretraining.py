import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_albert_pretraining import (
    ProcessorForAlbertPretraining,
    DataCollatorForAlbertPretraining,
)
from tests.transformers.tasks.test_data_base import TestDataBinarization

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataAlbertPretraining(TestDataBinarization):
    def __init__(
        self,
        model_name,
        parallel_context=None,
    ):
        self.processor = ProcessorForAlbertPretraining(model_name)
        self.data_collator = DataCollatorForAlbertPretraining(self.processor)
        if parallel_context is not None:
            self.sp_data_collator = DataCollatorForAlbertPretraining(
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
        label_pad_token_id=-100,
        batch_size=1024,
        batch_check_num_sample=2,
        batch_check_tokens=False,
        must_be_equal_to_max_length=True,
    ):
        self.processor._chunk_size = max_length - 3
        self.processor._max_length = max_length
        self.data_collator.mlm_probability = mlm_probability
        self.data_collator.label_pad_token_id = label_pad_token_id
        if self.sp_data_collator:
            self.sp_data_collator.mlm_probability = mlm_probability
            self.sp_data_collator.label_pad_token_id = label_pad_token_id

        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {max_length}",
            f"Batch size: {batch_size}",
            f"MLM probability: {mlm_probability}",
            sep="\n",
        )
        processed_dataset = dataset.map(
            self.processor, batched=True, remove_columns=dataset["train"].column_names
        )

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
            must_be_equal_to_max_length=must_be_equal_to_max_length,
        )

        self._length_check(
            dataloader,
            "labels",
            max_length,
            must_be_equal_to_max_length=must_be_equal_to_max_length,
        )

        self.mask_ratio_check(dataloader, label_pad_token_id)

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)

        print("---------- Test Pass ----------\n")

    def mask_ratio_check(self, dataloader, label_pad_token_id):
        mask_token_id = self.tokenizer.mask_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        for batch in dataloader:
            batch_size, seq_length = batch["input_ids"].size()

            # Verify that the mask token ratio is aligned to a predetermined percentage
            num_bos_tokens = torch.sum(batch["input_ids"] == bos_token_id)
            num_eos_tokens = torch.sum(batch["input_ids"] == eos_token_id)
            num_total = (batch_size * seq_length) - (num_bos_tokens + num_eos_tokens)
            mlm_probability = (
                torch.sum(batch["labels"] != label_pad_token_id) / num_total
            )
            assert torch.isclose(
                mlm_probability,
                torch.tensor(self.data_collator.mlm_probability),
                atol=0.002,
            ), f"Mask ratio({mlm_probability:.6f}) is different from the predefined one({self.data_collator.mlm_probability})"

            # Check that 20% of the mask tokens are not masked
            masked_tokens = batch["input_ids"][batch["labels"] != label_pad_token_id]
            random_word_probability = (
                torch.sum(masked_tokens != mask_token_id) / num_total
            )
            assert torch.isclose(
                random_word_probability,
                torch.tensor(self.data_collator.mlm_probability * 0.2),
                atol=0.002,
            ), f"Random word ratio({random_word_probability:.6f}) is different from the predefined one({(self.data_collator.mlm_probability*0.2)})"

        print(f"MLM Probability: {mlm_probability:.6f}")
        print("---- mask ratio test pass ----\n")


if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    # albert_test = TestDataAlbertPretraining("albert-base-v2")
    # albert_test(512, dataset)
    # albert_test(512, dataset, mlm_probability=0.3)
    # albert_test(128, dataset)

    parallel_context = ParallelContext.from_torch(sequence_parallel_size=4)
    albert_sp_test = TestDataAlbertPretraining("albert-base-v2", parallel_context)
    albert_sp_test(253, dataset, label_pad_token_id=0)
