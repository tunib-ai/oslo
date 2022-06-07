from typing import Optional
from torch.utils.data import DataLoader
import torch


class TestDataBinarization:
    def __init__(self, processor, data_collator, sp_data_collator):
        self.processor = processor
        self.data_collator = data_collator
        self.sp_data_collator = sp_data_collator
        self.tokenizer = data_collator.tokenizer

    def _batch_check(
        self, 
        batch, 
        num_samples: int = 2, 
        check_token: bool = False
        ) -> None:
        # Check batch after preprocessing and collection
        print("--------- batch check ---------\n")
        print(f"batch keys: {', '.join([key for key in batch.keys()])}\n")
        for key, value in batch.items():
            print(f"{key} size: {value.size()}")

        for idx in range(num_samples):
            for key, value in batch.items():
                print(f"{key}: \n{value[idx]}\n")

            for key, value in batch.items():
                if key == "input_ids":
                    print(f"input_ids decode: \n{self.tokenizer.decode(value[idx])}\n")
                elif key == "labels" and value.dim() != 1:
                    if torch.any(value[idx] < 0):
                        continue
                    print(f"labels decode: \n{self.tokenizer.decode(value[idx])}\n")
            
            if check_token:
                print(f"tokens: \n{self.tokenizer.convert_ids_to_tokens(batch['input_ids'][idx])}\n")
    
    def _length_check(
        self, 
        dataloader,
        key,
        max_length,
        pad_to_multiple_of: Optional[int],
        must_be_equal_to_max_length: bool = False
        ):
        for batch in dataloader:
            seq_length = batch[key].size(1)
            if must_be_equal_to_max_length:
                if pad_to_multiple_of is None:
                    assert (
                        seq_length == max_length
                    ), f"{key} sequence_length({seq_length}) must be equal to max_length({max_length})"
                else:
                    assert (
                        seq_length % pad_to_multiple_of == 0
                    ), f"{key} sequence length({seq_length}) must be equal to multiple of {pad_to_multiple_of}"
            else:
                if pad_to_multiple_of is None:
                    assert (
                        seq_length <= max_length
                    ), f"{key} sequence_length({seq_length}) must be shorter than max_length({max_length})"
                else:
                    assert (
                        seq_length % pad_to_multiple_of == 0
                    ), f"{key} sequence length({seq_length}) must be equal to multiple of {pad_to_multiple_of}"
        
        print(f"---- {key} length check test pass ----\n")
    
    def _test_sp_collator(self, processed_dataset, batch_size):
        local_world_size = self.sp_data_collator.local_world_size

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
            print("pad_token is set.")

        if self.sp_data_collator.tokenizer.pad_token is None:
            self.sp_data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
            print("pad_token is set. (SP)")

        dataloader = DataLoader(
            processed_dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=self.data_collator
        )
        sp_dataloader = DataLoader(
            processed_dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=self.sp_data_collator
        )
        
        for batch, sp_batch in zip(dataloader, sp_dataloader):
            seq_length = batch['input_ids'].size(1)
            sq_seq_length = sp_batch['input_ids'].size(1)
            
            sp_desired_length = ((seq_length // local_world_size) + 1)
            assert (
                sp_desired_length == sq_seq_length
            ), f"Required length for SP({sp_desired_length} doesn't equal to SP sequence length({sq_seq_length}))"

        print("---- SP collator test pass ----\n")