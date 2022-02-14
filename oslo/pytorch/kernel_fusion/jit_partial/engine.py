from contextlib import suppress

import torch

from oslo.pytorch.kernel_fusion.jit_partial.fused_gelu import fused_gelu


class JITPartialCompilingEngine(object):
    def __init__(self, model):
        self.model = model

    def fuse(self):
        self.fuse_activation(self.model)

    @staticmethod
    def fuse_activation(module):
        from transformers.activations import ACT2FN

        activations = list(ACT2FN.values())
        for name, child in module.named_modules():
            with suppress(Exception):
                for key, val in child.__dict__.items():
                    if val in activations:
                        if "gelu" in key.lower():
                            child.__dict__[key] = fused_gelu
                        else:
                            child.__dict__[key] = torch.jit.script(val)
