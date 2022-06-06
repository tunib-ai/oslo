from audioop import add
import torch
from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_t5_pretraining import (
    ProcessorForT5Pretraining,
    DataCollatorForT5Pretraining,
)
try:
    from datasets import load_dataset
except ImportError:
    print("You have to install 'datasets' to test data_sequence_classification.py")

# export PYTHONPATH="${PYTHONPATH}:/Users/gimmaru/oslo"

    
def test_t5_pretraining(model_name, max_length, dataset, mlm_probability=0.15, mean_noise_span_length=3):
    print(f"---- {model_name} test ----\n")
    processor = ProcessorForT5Pretraining(
        model_name, max_length, mlm_probability, mean_noise_span_length=mean_noise_span_length
    )
    
    processed_dataset = dataset.map(
        processor,
        batched=True,
        remove_columns=dataset['train'].column_names,
    )

    data_collator = DataCollatorForT5Pretraining(processor)
    dataloader = DataLoader(processed_dataset['train'], batch_size=1024, collate_fn=data_collator, shuffle=True)
    tokenizer = data_collator.tokenizer
    additional_special_ids = tokenizer.additional_special_tokens_ids
    # Check batch after preprocessing and collection
    print("---- batch check ----\n")
    batch = next(iter(dataloader))
    print(f"batch keys: {[key for key in batch.keys()]}\n")
    for key, value in batch.items():
        print(f"{key} size: {value.size()}")

    for idx in range(dataloader.batch_size):
        if idx == 2:
            break
        for key, value in batch.items():
            print(f"{key}: \n{value[idx]}\n")

        for key, value in batch.items():
            if key == "input_ids":
                print(f"input_ids decode: \n{tokenizer.decode(value[idx])}\nlength: {len(value[idx])}\n")
                input_ids = value[idx]
            elif key == "labels":
                print(f"labels decode: \n{tokenizer.decode(value[idx])}\nlength: {len(value[idx])}\n")
                labels = value[idx]
        
        text = []
        for input_id in input_ids:
            if input_id not in additional_special_ids:
                text.append(input_id)
            else:
                for label_idx, label_id in enumerate(labels):
                    if label_idx == 0:
                        continue
                    if label_id not in additional_special_ids:
                        text.append(label_id)
                    else:
                        labels = labels[label_idx:]
                        break
        
        print(f"text: \n{tokenizer.decode(text)}\nlength: {len(text)}\n")

    max_length = processor._max_length
    mean_noise_span_length = processor.mean_noise_span_length
    min_additional_special_id = min(additional_special_ids)
    multiple = lambda x, y: x * y
    for batch in dataloader:
        seq_length = batch['input_ids'].size(1)
        assert (seq_length == max_length), f"sequence_length({seq_length}) must be equal to max_length({max_length})"
        
        # Verify that the mask token is aligned to a predetermined percentage
        num_mask_span = torch.sum(batch['labels'] >= min_additional_special_id)
        mean_mask_tokens = num_mask_span * mean_noise_span_length
        num_input_ids = multiple(*batch['input_ids'].size())
        num_labels = multiple(*batch['labels'].size())
        total_num = num_input_ids + num_labels - 2*num_mask_span
        mlm_probability = torch.round(mean_mask_tokens / total_num, decimals=2)
        assert(processor.mlm_probability == mlm_probability), f"{mlm_probability} != {processor.mlm_probability}"
        
    print("---- test pass ----\n")

    return processor, processed_dataset

def test_sp_collator(
    processor,
    processed_datasets, 
    sequence_parallel_size: int = 3
    ):

    print("---- SP Collator test ----\n")
    parallel_context = ParallelContext.from_torch(sequence_parallel_size=sequence_parallel_size)
    data_collator = DataCollatorForT5Pretraining(processor)
    sp_data_collator = DataCollatorForT5Pretraining(processor, parallel_context=parallel_context)
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
    processor, processed_dataset = test_t5_pretraining("t5-small", 128, dataset)
    processor, processed_dataset = test_t5_pretraining("t5-small", 512, dataset)
    processor, processed_dataset = test_t5_pretraining("t5-small", 256, dataset, 0.2, 2)
    processor, processed_dataset = test_t5_pretraining("t5-small", 512, dataset, 0.3, 4)
    
    # test_sp_collator(processor, processed_dataset)
    # test_sp_collator(processor, processed_dataset)