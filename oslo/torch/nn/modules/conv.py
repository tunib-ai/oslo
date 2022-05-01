import torch
import torch.nn as nn
from torch.nn.modules.lazy import LazyModuleMixin


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al.
    for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
        skip_bias_add (`bool`): This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    References:
        https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
    """

    def __init__(self, nf, nx, skip_bias_add=False):
        super().__init__()
        self.nf = nf
        self.skip_bias_add = skip_bias_add

        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if not self.skip_bias_add:
            return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(
                size_out
            )
        else:
            return (
                torch.mm(x.view(-1, x.size(-1)), self.weight).view(size_out),
                self.bias,
            )


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
