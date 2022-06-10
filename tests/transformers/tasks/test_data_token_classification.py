from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_token_classification import (
    ProcessorForTokenClassification,
    DataCollatorForTokenClassification,
)
from tests.transformers.tasks.test_data_base import TestDataBinarization

try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")


class TestDataTokenClassification(TestDataBinarization):
    def __init__(
        self,
        model_name,
        dataset,
        parallel_context=None,
    ):
        self.processor = ProcessorForTokenClassification(model_name, dataset=dataset)
        self.data_collator = DataCollatorForTokenClassification(self.processor)
        self.sp_data_collator = DataCollatorForTokenClassification(
            self.processor, parallel_context=parallel_context
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.parallel_context = parallel_context
        self.dataset = dataset

    def __call__(
        self,
        max_length,
        batch_size=64,
        pad_to_multiple_of=None,
        batch_check_num_sample=2,
        batch_check_tokens=False,
        must_be_equal_to_max_length=False,
        stop_idx=10,
    ):
        self.processor._chunk_size = max_length
        self.processor._max_length = max_length
        self.data_collator.pad_to_multiple_of = pad_to_multiple_of
        dataset = self.dataset

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

        self.token_to_label_check(batch, batch_size, stop_idx)

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)

        print("---------- Test Pass ----------\n")

    def token_to_label_check(self, batch, batch_size, stop_idx):
        label_map = self.processor.get_label_map(self.processor.label_names)
        for idx in range(batch_size):
            if idx == stop_idx:
                break
            print("---- Verify Tokens and Labels are Correctly Corresponded ----")
            tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][idx])
            labels = [
                label_map["id2label"][str(label)] if label != -100 else "[NONE]"
                for label in batch["labels"][idx].numpy()
            ]

            token_to_label = [(token, label) for token, label in zip(tokens, labels)]
            print(token_to_label, end="\n")


if "__main__" == __name__:
    dataset = load_dataset("klue", "ner")
    dataset = dataset.rename_column("ner_tags", "labels")

    # klue ner
    gpt2_test = TestDataTokenClassification("skt/kogpt2-base-v2", dataset)
    gpt2_test(512, 128)
    gpt2_test(256, 128, 3)
    gpt2_test(32, 64)

    bert_test = TestDataTokenClassification("klue/bert-base", dataset)
    bert_test(512, 16)
    gpt2_test(256, 128, 3)
    bert_test(32, 64)

    roberta_test = TestDataTokenClassification("klue/roberta-base", dataset)
    roberta_test(512, 1024)
    gpt2_test(256, 128, 3)
    roberta_test(32, 64)

    # conll2003 ner pos
    dataset = load_dataset("conll2003")
    conll_ner_dataset = dataset.rename_column("ner_tags", "labels")
    conll_pos_dataset = dataset.rename_column("pos_tags", "labels")

    bert_conll_ner_test = TestDataTokenClassification(
        "bert-base-cased", conll_ner_dataset
    )
    bert_conll_ner_test(32, 64)
    bert_conll_pos_test = TestDataTokenClassification(
        "bert-base-cased", conll_pos_dataset
    )
    bert_conll_pos_test(32, 64)

    # parallel_context = ParallelContext.from_torch(sequence_parallel_size=3)
    # bert_sp_test = TestDataTokenClassification("bert-base-cased", conll_ner_dataset, parallel_context)
    # bert_sp_test(256, dataset, 1024)
