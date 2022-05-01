import torch
import math
from torch.utils.data import default_collate
from oslo.torch.distributed import ParallelContext, ParallelMode
from typing import Optional, Callable, List

class SequenceParallelDataCollatorWrapper:
    def __init__(
        self,
        parallel_context: ParallelContext,
        parallelize_keys: List[str],
        batch_first: bool = True,
        pad_token_id: Optional[int] = 0,
        collate_fn: Optional[Callable] = None
    ):
        self.parallelize_keys = parallelize_keys
        self.batch_index = 0 if batch_first else 1
        self.seq_index = 1 if batch_first else 0
        self.pad_token_id = pad_token_id
        self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
        self.collate_fn = collate_fn if collate_fn is not None else default_collate

    def __call__(self, **kwargs):

        features = self.collate_fn(**kwargs)
        for key in self.parallelize_keys:
            value = features[key]
            batch_size = value.size(self.batch_index)
            seq_length = value.size(self.seq_index)
            remainder = seq_length % self.local_world_size
            if remainder > 0:
                pads = torch.ones(size=(batch_size, remainder), dtype=value.dtype, device=value.device)
                value = torch.hstack((value, pads))
            sub_seq_length = math.ceil(seq_length / self.local_world_size)
            start_idx = self.local_rank * sub_seq_length
            end_idx = (self.local_rank + 1) * sub_seq_length
            features[key] = value[:, start_idx:end_idx]

        return features
