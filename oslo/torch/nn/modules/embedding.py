import torch
import torch.nn as nn
import torch.nn.functional as F

from oslo.torch.distributed import ParallelContext, ParallelMode


class VocabUtility:
    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        assert (
            global_vocab_size % world_size == 0
        ), "vocab size must be divisible by world size"

        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank
        )




class VocabParallelEmbedding1D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        parallel_context: ParallelContext,
    ):
        self.parallel_context = parallel_context
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings, rank, world_size
        )

        super().__init__(
            num_embeddings=num_embeddings // world_size,
            embedding_dim=embedding_dim,
            device=torch.device(torch.cuda.current_device()),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_reduce_1d,
        )

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)

        if world_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if world_size > 1:
            output_parallel[input_mask, :] = 0.0

        # Reduce across all the model parallel GPUs.
        output = all_reduce_1d(output_parallel, self.parallel_context)
        return output


class Embedding2D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        parallel_context: ParallelContext,
    ):
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.summa_dim**2),
            device=torch.device(torch.cuda.current_device()),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
            all_gather_tensor_2d,
            # split_batch_2d,
        )

        # input_ = split_batch_2d(input_, self.parallel_context)
        weight = all_gather_tensor_2d(
            self.weight, -1, ParallelMode.TENSOR_2D_COL, self.parallel_context
        )
        output = F.embedding(
            input_,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output


class VocabParallelEmbedding2D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        parallel_context: ParallelContext,
    ):
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2D_COL)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            rank,
            self.summa_dim,
        )
        super().__init__(
            num_embeddings=num_embeddings // self.summa_dim,
            embedding_dim=embedding_dim // self.summa_dim,
            device=torch.device(torch.cuda.current_device()),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
        #     reduce_scatter_tensor_2d,
        # )
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        output = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output[input_mask, :] = 0.0
        # output = reduce_scatter_tensor_2d(
        #     output_parallel, 0, ParallelMode.TENSOR_2D_COL, self.parallel_context
        # )
        return output


class Embedding2p5D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        parallel_context: ParallelContext,
    ):
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.tesseract_dim**2),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            all_gather_tensor_2p5d,
            split_batch_2p5d,
        )

        input_ = split_batch_2p5d(input_, 0, self.parallel_context)
        weight = all_gather_tensor_2p5d(
            self.weight, -1, ParallelMode.TENSOR_2P5D_COL, self.parallel_context
        )

        # output = F.embedding(input_, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)
        output = F.embedding(
            input_,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output


# TODO: Implement this class.
class VocabParallelEmbedding2p5D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        parallel_context: ParallelContext,
    ):
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )

        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            num_embeddings,
            rank,
            self.tesseract_dim,
        )
        super().__init__(
            num_embeddings=num_embeddings // self.tesseract_dim,
            embedding_dim=embedding_dim // self.tesseract_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            reduce_scatter_tensor_2p5d,
        )

        # Build the mask.
        input_mask = (x < self.vocab_start_index) | (x >= self.vocab_end_index)
        # Mask the input.
        masked_input = x.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_scatter_tensor_2p5d(
            output_parallel, 0, ParallelMode.TENSOR_2P5D_COL, self.parallel_context
        )
        return output
