from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")

# export PYTHONPATH="${PYTHONPATH}:/Users/gimmaru/oslo"


# parallel_context = ParallelContext.from_torch(sequence_parallel_size=3)
    
def test_processor(model_name, max_length, dataset):
        print(f"---- {model_name} test ----")
        processor = ProcessorForSequenceClassification(model_name, max_length)
        processed_dataset = dataset.map(
            processor,
            batched=True,
            remove_columns = dataset['train'].column_names
        )
        processed_dataset.cleanup_cache_files()
        max_length = processor._max_length
        for data_purpose in processed_dataset.keys():
            purposed_dataset = processed_dataset[data_purpose] 
            print(f"{data_purpose} dataset columns: {purposed_dataset.column_names}")
            for key in purposed_dataset.features:
                if key == "input_ids":
                    for value in purposed_dataset[key]:
                        assert isinstance(value, list), f"input_ids column value must be List, but value is {value}"
                        assert (len(value) <= max_length), f"{data_purpose} dataset {key} length exceed max_length({max_length})"
                if key == "labels":
                    for value in purposed_dataset[key]:
                        assert isinstance(value, int), f"labels column value must be int, but value is {value}"
        print("---- processor test pass ----")

        return processor, processed_dataset

def test_collator(processor, processed_datasets, batch_size, pad_to_multiple_of=None):
    max_length = processor._max_length
    data_collator = DataCollatorForSequenceClassification(processor, pad_to_multiple_of=pad_to_multiple_of)
    
    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        print("pad_token is set.")

    train_dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(processed_datasets['validation'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(processed_datasets['test'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    
    tokenizer = data_collator.tokenizer
    print("---- batch check ----")
    batch = next(iter(train_dataloader))
    print(f"batch keys: {[key for key in batch.keys()]}")
    for idx in range(train_dataloader.batch_size):
        if idx == 3:
            break
        for key, value in batch.items():
            print(f"{key}: \n{value[idx]}")
            if key == "input_ids":
                print(f"tokens: \n{tokenizer.convert_ids_to_tokens(value[idx])}")

    dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    for dataloader in dataloaders:
        for batch in dataloader:
            for key, value in batch.items():
                if key == "input_ids":
                    seq_length = value.size(1)
                    if pad_to_multiple_of is None:
                        assert (seq_length <= max_length), f"input_ids sequence length({seq_length}) must be shorter than the max_length({max_length})"
                    else:
                        assert (seq_length % pad_to_multiple_of == 0), f"input_ids sequence length({seq_length}) must be equal to multiple of {pad_to_multiple_of}"
                if key == "labels":
                    assert value.dim() == 1, f"labels dimension must be 1"
    print("---- collator test pass ----")

def test_sp_collator(processor, processed_datasets, batch_size, parallel_context=None):
    data_collator = DataCollatorForSequenceClassification(processor)
    sp_data_collator = DataCollatorForSequenceClassification(processor, parallel_context=parallel_context)
    local_world_size = sp_data_collator.local_world_size

    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        sp_data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        print("pad_token is set.")

    dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    sp_dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=sp_data_collator)

    print(f"---- SP batch check ----\n")
    sp_batch = next(iter(sp_dataloader))
    for key, value in sp_batch.items():
        print(f"{key} size: {value.size()}")
        print(f"{key} sample: \n{value[0]}\n")

    for batch, sp_batch in zip(dataloader, sp_dataloader):
        seq_length = batch['input_ids'].size(1)
        sq_seq_length = sp_batch['input_ids'].size(1)
        
        if seq_length % local_world_size != 0:
            sp_desired_length = ((seq_length // local_world_size) + 1)
        else:
            sp_desired_length = seq_length

        assert (sp_desired_length == sq_seq_length), f"Required length for SP({sp_desired_length} doesn't equal to SP sequence length({sq_seq_length}))"
            
def sequence_classification_test(
    model_name,
    max_length, 
    dataset,
    batch_size,
    pad_to_multiple_of=None,
):
    processor, processed_dataset = test_processor(
        model_name=model_name, 
        max_length=max_length, 
        dataset=dataset,
    )
    test_collator(
        processor=processor,
        processed_datasets=processed_dataset,
        batch_size=batch_size,
        pad_to_multiple_of=pad_to_multiple_of,
    )

if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("label", "labels")
    sequence_classification_test("gpt2", 256, dataset, 1024, 3)
    sequence_classification_test("gpt2", 512, dataset, 1024)
    sequence_classification_test("bert-base-cased", 64, dataset, 64, 3)
    sequence_classification_test("roberta-base", 64, dataset, 512, 4)
    sequence_classification_test("albert-base-v2", 64, dataset, 128)
    sequence_classification_test("facebook/bart-base", 64, dataset, 32)
    sequence_classification_test("t5-small", 64, dataset, 1024)