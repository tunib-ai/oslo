# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from .functional import fused_scale_mask_softmax


class FusedScaleMaskSoftmax(nn.Module):
    """
    Kernel fusion function: Scale + Mask + Softmax
    Before fusion: Matmul (Q * K^T) -> Scale -> Mask -> Softmax -> Matmul (Score * V) -> ...
    After fusion: Matmul (Q * K^T) -> (Scale + Mask + Softmax) -> Matmul (Score * V) -> ...
    """

    def __init__(self, scale, use_triang_mask, softmax_in_fp32=True, pad_mask=None):
        self.scale = scale
        self.use_triang_mask = use_triang_mask
        self.pad_mask = pad_mask
        self.softmax_in_fp32 = softmax_in_fp32
        super(FusedScaleMaskSoftmax, self).__init__()

    def forward(self, input):
        return fused_scale_mask_softmax(
            input, self.scale, self.use_triang_mask, self.softmax_in_fp32, self.pad_mask
        )
