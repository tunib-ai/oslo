import torch
import torch.nn as nn

from oslo.pytorch.kernel_fusion.compile.aot_autograd import aot_function
from oslo.pytorch.kernel_fusion.compile.compat import _stateless
from oslo.pytorch.kernel_fusion.compile.compilers import ts_compile

input_data = torch.randn(4, 4).cuda()
l1 = nn.Linear(4, 4).cuda()
l2 = nn.Linear(4, 4).cuda()
l1_trace = aot_function(l1.forward, fw_compiler=ts_compile)

with _stateless.reparametrize_module(l1, dict(l2.named_parameters())):
    print(l1_trace(input_data))

print(l2(input_data))
