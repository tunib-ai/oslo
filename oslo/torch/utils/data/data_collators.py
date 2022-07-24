from typing import List, Optional

import torch

from oslo.torch.distributed import ParallelContext, ParallelMode


class SequenceDataParallelCollator:
    def __init__(
        self,
        tokenizer,
        parallel_keys: List[str],
        parallel_context: ParallelContext,
        dim: int = 1,
    ):
        self.parallel_keys = parallel_keys
        self.dim = dim
        self.tokenizer = tokenizer
        self.local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
        self.local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)

    def __call__(self, **features):
        for key in self.parallel_keys:
            assert (
                key in features
            ), f"The {key} must be in the input of `SequenceDataParallelCollator`."

            value = features[key]
            value_size = value.size()
            seq_length = value_size[self.dim]

            new_seq_length = seq_length
            while new_seq_length % self.local_world_size != 0:
                new_seq_length += 1
            num_pads = new_seq_length - seq_length

            if num_pads > 0:
                pad_size = list(value_size)
                pad_size[self.dim] = num_pads

                if key in ["input_ids", "decoder_input_ids"]:
                    pad_token_id = self.tokenizer.pad_token_id
                elif key in ["attention_mask", "decoder_attention_mask"]:
                    pad_token_id = 0
                elif key == "token_type_ids":
                    pad_token_id = self.tokenizer.pad_token_type_id

                pads = (
                    torch.ones(
                        pad_size,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    * pad_token_id
                )
                value = torch.cat([value, pads], dim=self.dim)

            value = value.chunk(
                self.local_world_size,
                dim=self.dim,
            )[self.local_rank]

            if not value.is_contiguous():
                value = value.contiguous()

            features[key] = value

        return features
