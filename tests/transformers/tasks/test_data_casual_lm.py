import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
    DataCollatorForCausalLM,
)
try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")

# export PYTHONPATH="${PYTHONPATH}:/Users/gimmaru/oslo"


# parallel_context = ParallelContext.from_torch(sequence_parallel_size=3)
    
def test_processor(model_name, max_length, dataset):
        print(f"---- {model_name} test ----")
        processor = ProcessorForCausalLM(model_name, max_length)
        processed_dataset = dataset.map(
            processor,
            batched=True,
            remove_columns = dataset['train'].column_names
        )
        processed_dataset.cleanup_cache_files()
        max_length = processor._max_length
        for data_purpose in processed_dataset.keys():
            purposed_dataset = processed_dataset[data_purpose] 
            for value in purposed_dataset["input_ids"]:
                assert isinstance(value, list), f"input_ids column value must be List, but value is {value}"
                assert (len(value) == max_length), f"input_ids column value length must be equal to max_length({max_length})"
        print("---- processor test pass ----")

        return processor, processed_dataset

def test_collator(processor, processed_datasets, batch_size, pad_to_multiple_of=None):
    max_length = processor._max_length
    data_collator = DataCollatorForCausalLM(processor, pad_to_multiple_of=pad_to_multiple_of)
    
    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        print("pad_token is set.")

    train_dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(processed_datasets['validation'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(processed_datasets['test'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    tokenizer = data_collator.tokenizer
    print("---- sample check ----")
    sample = next(iter(train_dataloader))
    print(f"sample keys: {[key for key in sample.keys()]}")
    for idx in range(train_dataloader.batch_size):
        if idx == 3:
            break
        for key, value in sample.items():
            print(f"{key}: \n{value[idx]}")
            if key == "input_ids":
                print(f"tokens: \n{tokenizer.convert_ids_to_tokens(value[idx])}")

    dataloaders = [train_dataloader, valid_dataloader, test_dataloader]
    for dataloader in dataloaders:
        for batch in dataloader:
            assert (torch.all(batch["input_ids"].eq(batch["labels"]))), f"input_ids and labels must be the same."
            seq_length = batch["input_ids"].size(1)
            if pad_to_multiple_of is None:
                assert (seq_length == max_length), f"input_ids sequence length({seq_length}) must be equal to the max_length({max_length})"
            else:
                assert (seq_length % pad_to_multiple_of == 0), f"input_ids sequence length({seq_length}) must be equal to multiple of {pad_to_multiple_of}"
           
    print("---- collator test pass ----")

def test_sp_collator(processor, processed_datasets, batch_size, parallel_context=None):
    data_collator = DataCollatorForCausalLM(processor)
    sp_data_collator = DataCollatorForCausalLM(processor, parallel_context=parallel_context)
    local_world_size = sp_data_collator.local_world_size

    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        sp_data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        print("pad_token is set.")

    dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    sp_dataloader = DataLoader(processed_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=sp_data_collator)
    
    for batch, sp_batch in zip(dataloader, sp_dataloader):
        seq_length = batch['input_ids'].size(1)
        sq_seq_length = sp_batch['input_ids'].size(1)
        
        sp_desired_length = ((seq_length // local_world_size) + 1)
        assert (sp_desired_length == sq_seq_length), f"Required length for SP({sp_desired_length} doesn't equal to SP sequence length({sq_seq_length}))"
            
def test_causal_lm(
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
    test_causal_lm("gpt2", 256, dataset, 1024, 3)
    test_causal_lm("gpt2", 512, dataset, 1024)
    test_causal_lm("facebook/bart-base", 64, dataset, 32)
    test_causal_lm("t5-small", 64, dataset, 1024)