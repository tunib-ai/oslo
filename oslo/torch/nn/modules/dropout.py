import torch
from torch.nn.modules.dropout import _DropoutNd

from oslo.torch.nn.modules.functional import (
    fused_bias_dropout,
    fused_bias_dropout_residual,
)


class FusedBiasDropout(_DropoutNd):
    def forward(self, input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return fused_bias_dropout(input, bias, self.p, self.training, self.inplace)
