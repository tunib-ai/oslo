from typing import Optional

import torch
import itertools
import torch.nn as nn
from torch.nn import UninitializedParameter
from torch.nn.parameter import is_lazy
from torch.nn.modules.lazy import _LazyProtocol

from oslo.torch.nn import Conv1D, Linear


class LazyModuleMixin:
    cls_to_become = None

    def __init__(self: _LazyProtocol, *args, **kwargs):
        # Mypy doesn't like this super call in a mixin
        super().__init__(*args, **kwargs)  # type: ignore[misc]

    def has_uninitialized_params(self: _LazyProtocol):
        r"""Check if a module has parameters that are not initialized"""
        # This is to avoid the JIT to track this parameter and force
        # custom modules __setstate__ to add it
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True
        return False

    def initialize_parameters(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, LazyModuleMixin):
                module.initialize_parameters(*args, **kwargs)


class LazyConv1D(LazyModuleMixin, Conv1D):
    """
    Lazy initialized Conv1D layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of parameters are copied to the CPU RAM, which causes the CPU RAM to run out.
    Therefore, after creating uninitialized parameters and re-adjusting them to a suitable size,
    you can initialize only the necessary parameters to a suitable GPU immediately.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
        skip_bias_add (`bool`): This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.

    Examples:
        >>> from oslo.torch.nn import LazyConv1D

        >>> layer = LazyConv1D(2, 4)
        >>> print(layer.weight)
        <UninitializedParameter>

        >>> layer.initialize_parameters()
        >>> print(layer.weight)
        Parameter containing:
        tensor([[ 0.0293,  0.0119,  0.0055,  0.0132],
                [-0.0578,  0.0084, -0.0180, -0.0174]], requires_grad=True)

    References:
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
    """

    cls_to_become = Conv1D
    weight: nn.UninitializedParameter
    bias: nn.UninitializedParameter

    def __init__(self, nx: int, nf: int, skip_bias_add: bool = False) -> None:
        super().__init__(0, 0, skip_bias_add=skip_bias_add)
        self.nx = nx
        self.nf = nf
        self.weight = nn.UninitializedParameter(device=None, dtype=None)
        self.bias = nn.UninitializedParameter(device=None, dtype=None)

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.nx, self.nf))
                nn.init.normal_(self.weight, std=0.02)
                if self.bias is not None:
                    self.bias.materialize((self.nf,))
                    self.bias.zero_()


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


class LazyLinear(LazyModuleMixin, Linear):
    """
    Lazy initialized linear layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of parameters are copied to the CPU RAM, which causes the CPU RAM to run out.
    Therefore, after creating uninitialized parameters and re-adjusting them to a suitable size,
    you can initialize only the necessary parameters to a suitable GPU immediately.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.

    Notes:
        This is different from torch.nn.LazyLinear in terms of
        ``in_features`` can be input by user at the creation time.

    Examples:
        >>> from oslo.torch.nn import LazyLinear

        >>> layer = LazyLinear(2, 4)
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

    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            skip_bias_add: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(0, 0, False)
        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        self.weight = UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.cls_to_become is not None:
            self.__class__ = self.cls_to_become
