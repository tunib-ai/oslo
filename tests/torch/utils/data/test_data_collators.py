import torch

from oslo.torch.distributed import ParallelContext
from oslo.torch.utils.data import SequenceDataParallelCollator

parallel_context = ParallelContext.from_torch(sequence_parallel_size=4)

data = {
    "input_ids": torch.randn(16, 129).cuda(),
    "attention_mask": torch.ones(16, 129).cuda(),
}

collator = SequenceDataParallelCollator(
    parallel_context=parallel_context,
    parallel_keys=["input_ids", "attention_mask"],
    pad_token_id=99,
)

sharded = collator(**data)
print(sharded["input_ids"].size())
