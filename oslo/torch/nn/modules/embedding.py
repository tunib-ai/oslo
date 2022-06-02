from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import UninitializedParameter

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.modules.lazy import LazyModuleMixin


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


class LazyEmbedding(LazyModuleMixin, nn.Embedding):
    """
    Lazy initialized embedding layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of parameters are copied to the CPU RAM, which causes the CPU RAM to run out.
    Therefore, after creating uninitialized parameters and re-adjusting them to a suitable size,
    you can initialize only the necessary parameters to a suitable GPU immediately.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                     the embedding vector at :attr:`padding_idx` will default to all zeros,
                                     but can be updated to another value to be used as the padding vector.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

    Notes:
        This is different from torch.nn.LazyLinear in terms of
        ``in_features`` can be input by user at the creation time.

    Examples:
        >>> from oslo.torch.nn import LazyEmbedding

        >>> layer = LazyEmbedding(4, 2)
        >>> print(layer.weight)
        <UninitializedParameter>

        >>> layer.initialize_parameters()
        >>> print(layer.weight)
        Parameter containing:
        tensor([[-0.7025,  0.5608],
                [-0.2529, -0.2636],
                [-0.5755, -0.2422],
                [ 0.4704,  0.6281]], requires_grad=True)
    """

    cls_to_become = nn.Embedding
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LazyEmbedding, self).__init__(
            num_embeddings=0,
            embedding_dim=0,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_embeddings != 0:
            super().reset_parameters()

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.num_embeddings, self.embedding_dim))
                self.reset_parameters()
        if self.cls_to_become is not None:
            self.__class__ = self.cls_to_become


class VocabParallelEmbedding1D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
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
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_1d._ops import (
            all_reduce_1d,
        )

        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)

        if world_size > 1:
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

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
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.summa_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2D_COL
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.summa_dim**2),
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
            all_gather_tensor_2d,
        )
        weight = all_gather_tensor_2d(
            self.weight, -1, self.parallel_context, ParallelMode.TENSOR_2D_COL,
        )
        output = F.embedding(
            input,
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
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
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
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2d._ops import (
            reduce_by_batch_2d,
        )
        world_size = self.parallel_context.get_world_size(ParallelMode.TENSOR)
        if world_size > 1:
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input
            
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
        output = reduce_by_batch_2d(
            output, 
            False,
            parallel_context=self.parallel_context,
        )
        return output


class Embedding2p5D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.tesseract_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_2P5D_COL
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // (self.tesseract_dim**2),
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_2p5d._ops import (
            all_gather_tensor_2p5d,
            split_batch_2p5d,
        )

        input = split_batch_2p5d(input, 0, self.parallel_context)
        weight = all_gather_tensor_2p5d(
            self.weight, -1, self.parallel_context, ParallelMode.TENSOR_2P5D_COL,
        )

        output = F.embedding(
            input,
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
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
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
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
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
            output_parallel, 0, self.parallel_context, ParallelMode.TENSOR_2P5D_COL
        )
        return output


class Embedding3D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.cubic_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_3D_INPUT,
        )

        self.input_parallel_mode = ParallelMode.TENSOR_3D_INPUT
        self.weight_parallel_mode = ParallelMode.TENSOR_3D_WEIGHT
        self.output_parallel_mode = ParallelMode.TENSOR_3D_OUTPUT

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim // self.cubic_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import (
            split_tensor_3d,
            broadcast_weight_3d_from_diagonal,
        )
        input = split_tensor_3d(input, 0, self.parallel_context, self.weight_parallel_mode)
        input = split_tensor_3d(input, 0, self.parallel_context, self.input_parallel_mode)
        weight = broadcast_weight_3d_from_diagonal(
            self.weight, 
            self.parallel_context,
            self.input_parallel_mode, 
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )
        output = F.embedding(
            input, 
            weight, 
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return output


class VocabParallelEmbedding3D(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        parallel_context: Optional[ParallelContext] = None,
    ):
        self.parallel_context = parallel_context
        self.cubic_dim = self.parallel_context.get_world_size(
            ParallelMode.TENSOR_3D_INPUT,
        )

        self.input_parallel_mode = ParallelMode.TENSOR_3D_INPUT
        self.weight_parallel_mode = ParallelMode.TENSOR_3D_WEIGHT
        self.output_parallel_mode = ParallelMode.TENSOR_3D_OUTPUT

        vocab_parallel_rank = self.parallel_context.get_local_rank(self.input_parallel_mode)

        super().__init__(
            num_embeddings=num_embeddings // (self.cubic_dim ** 2),
            embedding_dim=embedding_dim // self.cubic_dim,
            device=torch.device(torch.cuda.current_device()),
            dtype=dtype,
        )
        self.vocab_start_index = vocab_parallel_rank * self.num_embeddings // self.cubic_dim
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings // self.cubic_dim

    def forward(self, input: Tensor) -> Tensor:
        from oslo.torch.nn.parallel.tensor_parallel._parallel_3d._ops import (
            split_tensor_3d,
            all_gather_tensor_3d,
            reduce_scatter_tensor_3d,
        )
        input = split_tensor_3d(input, 0, self.parallel_context, self.weight_parallel_mode)

        input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
        masked_input = input.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        weight = all_gather_tensor_3d(
            self.weight, 
            0, 
            self.parallel_context, 
            self.weight_parallel_mode,
        )

        output_parallel = F.embedding(
            masked_input, 
            weight, 
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        output_parallel[input_mask, :] = 0.
        output = reduce_scatter_tensor_3d(
            output_parallel, 
            0, 
            self.parallel_context, 
            self.input_parallel_mode,
        )

        return output
