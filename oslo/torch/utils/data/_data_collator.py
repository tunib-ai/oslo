import torch
from oslo.torch.distributed import ParallelContext, ParallelMode


class SequenceParallelDataCollator:
    def __init__(self, gpc: ParallelContext):
        self.local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        self.local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

    def __call__(self, features):
        if not isinstance(features[0], dict):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                seq_length = len(v)
                assert seq_length % self.local_world_size == 0, \
                    f'Sequence length must be multiply of world_size({self.local_world_size})'
                sub_seq_length = seq_length // self.local_world_size

                start_idx = self.local_rank * sub_seq_length
                end_idx = (self.local_rank + 1) * sub_seq_length
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k][start_idx:end_idx] for f in features])
                else:
                    dtype = torch.long if isinstance(v, int) else torch.float
                    batch[k] = torch.tensor([f[k][start_idx:end_idx] for f in features], dtype=dtype)

        return batch
