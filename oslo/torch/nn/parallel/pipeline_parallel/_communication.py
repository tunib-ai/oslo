import inspect

import torch.nn as nn

from oslo.torch.distributed import ParallelMode

"""
forward: pre-communication => forward => post-communication
backward: pre-communication <= forward <= post-communication
"""


def add_hook(module, parallel_context):
    forward = module.forward

    def new_forward(*args, **kwargs):
        frame = inspect.currentframe()
        while hasattr(frame, "f_back"):
            f_locals = frame.f_locals
            if "self" in f_locals and isinstance(f_locals["self"], nn.Module):
                break
            else:
                frame = frame.f_back

        caller_module = frame.f_locals["self"]
        if hasattr(caller_module, "oslo_parallel"):
            caller_rank = caller_module.oslo_parallel[ParallelMode.PIPELINE]
        else:
            caller_rank = None

        output = forward(*args, **kwargs)

    module.forward = new_forward
