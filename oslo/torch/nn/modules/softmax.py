from typing import Optional

import torch.nn as nn
import torch


class FusedScaleMaskSoftmaxFunction(torch.autograd.Function):
    pass


class FusedScaleMaskSoftmax(nn.Module):
    def __init__(self, dim: Optional[int] = None):
        super().__init__()

    def forward(self):
        pass
