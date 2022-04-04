from oslo.torch.nn.modules.functional import (
    fused_gelu,
    fused_bias_gelu,
    fused_bias_dropout,
    fused_bias_dropout_residual,
)
from oslo.torch.nn.modules.dropout import (
    FusedBiasDropout,
    FusedBiasDropoutResidual,
)
from oslo.torch.nn.modules.linear import LazyLinear
from oslo.torch.nn.modules.conv import Conv1D, LazyConv1D
