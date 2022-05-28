import torch
import torch.nn as nn


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


