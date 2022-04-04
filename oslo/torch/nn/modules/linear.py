from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.lazy import LazyModuleMixin


class LazyLinear(LazyModuleMixin, nn.Linear):
    """
    Lazy initialized linear layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of paramters are copied to the CPU RAM, which causes the CPU RAM to run out.
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

    cls_to_become = nn.Linear
    weight: nn.UninitializedParameter
    bias: nn.UninitializedParameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        skip_bias_add=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(0, 0, False)
        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        self.weight = nn.UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = nn.UninitializedParameter(**factory_kwargs)

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.skip_bias_add:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight), self.bias

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params():
            super().reset_parameters()

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


class ColumnParallelLinear(nn.Module):
    pass


class RowParallelLinear(nn.Module):
    pass


class Linear2D(nn.Module):
    pass
