import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_roberta_pretraining import (
    ProcessorForRobertaPretraining,
    DataCollatorForRobertaPretraining,
)
try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")

# export PYTHONPATH="${PYTHONPATH}:/Users/gimmaru/oslo"

    
def test_albert_pretraining(model_name, max_length, dataset, mlm_probability=0.15):
    print(f"---- {model_name} test ----\n")
    processor = ProcessorForRobertaPretraining(model_name, max_length)
    
    processed_dataset = dataset.map(
        processor,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )

    data_collator = DataCollatorForRobertaPretraining(processor, mlm_probability=mlm_probability)
    dataloader = DataLoader(processed_dataset['train'], batch_size=1024, collate_fn=data_collator, shuffle=True)
    tokenizer = data_collator.tokenizer
    
    # Check batch after preprocessing and collection
    print("---- batch check ----\n")
    batch = next(iter(dataloader))
    print(f"batch keys: {[key for key in batch.keys()]}\n")
    for idx in range(dataloader.batch_size):
        if idx == 3:
            break
        for key, value in batch.items():
            if key == "input_ids":
                print(f"tokens: \n{tokenizer.convert_ids_to_tokens(value[idx])}")
            print(f"{key}: \n{value[idx]}")

    max_length = processor._max_length
    mask_token_id = tokenizer.mask_token_id
    for batch in dataloader:
        batch_size, seq_length = batch['labels'].size()
        assert (seq_length == max_length), f"sequence_length({seq_length}) must be equal to max_length({max_length})"

        # Verify that the mask token is aligned to a predetermined percentage
        mlm_probability = torch.round(torch.sum(batch['labels'] != -100) / (batch_size*seq_length), decimals=2)
        assert(data_collator.mlm_probability == mlm_probability), f"{mlm_probability} != {data_collator.mlm_probability}"
        # assert torch.all((batch['input_ids'] == mask_token_id).eq(batch['labels'] != -100)), f"input_ids and labels matching false"

        # Check that 20% of the mask tokens are not masked
        masked_tokens = batch["input_ids"][batch["labels"] != -100]
        random_word_probability = torch.round(torch.sum(masked_tokens != mask_token_id) / (batch_size*seq_length), decimals=2)
        assert((data_collator.mlm_probability*0.2) == random_word_probability), f"{random_word_probability} != {(data_collator.mlm_probability*0.2)}"
    print("---- test pass ----\n")

    return processor, processed_dataset

def test_sp_collator(
    processor,
    processed_datasets, 
    sequence_parallel_size: int = 3
    ):

    print("---- SP Collator test ----\n")
    parallel_context = ParallelContext.from_torch(sequence_parallel_size=sequence_parallel_size)
    data_collator = DataCollatorForRobertaPretraining(processor)
    sp_data_collator = DataCollatorForRobertaPretraining(processor, parallel_context=parallel_context)
    local_world_size = sp_data_collator.local_world_size

    if data_collator.tokenizer.pad_token is None:
        data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        sp_data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
        print("pad_token is set now.")

    dataloader = DataLoader(processed_datasets['train'], batch_size=8, shuffle=False, collate_fn=data_collator)
    sp_dataloader = DataLoader(processed_datasets['train'], batch_size=8, shuffle=False, collate_fn=sp_data_collator)

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

    print("---- test pass ----\n")

if "__main__" == __name__:
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")
    processor, processed_dataset = test_albert_pretraining("roberta-base", 512, dataset, 0.3)
    processor, processed_dataset = test_albert_pretraining("roberta-base", 512, dataset, 0.15)
    # test_sp_collator(processor, processed_dataset)
    # processor, processed_dataset = test_albert_pretraining("klue/bert-base", 512, dataset)
    # test_sp_collator(processor, processed_dataset)