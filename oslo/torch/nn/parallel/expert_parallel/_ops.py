from abc import ABC

import torch
import torch.distributed as dist

from torch import Tensor
from typing import Any, Tuple, Optional

from torch.distributed import ProcessGroup

OSLO_EP_KERNEL_FLAG = False
try:
    import oslo_expert_parallel_cuda

    OSLO_EP_KERNEL_FLAG = True
except ImportError:
    print(
        "If you want to activate cuda kernel for Expert Parallel, Please install with cuda_extension."
    )


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any, inputs: Tensor, group: Optional[ProcessGroup] = None
    ) -> Tensor:
        if context is not None:
            context.comm_group = group

        if not inputs.is_contiguous():
            inputs = inputs.is_contiguous()

        if dist.get_world_size() == 1:
            return inputs

        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs, group=group)

        return output

    @staticmethod
    def backward(context: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        return AllToAll.forward(None, *grad_outputs, context.comm_group), None


class EPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(context, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)

        expert_input = oslo_expert_parallel_cuda.dispatch_forward(
            s, ec, h, tokens, mask, dest_idx
        )
        context.save_for_backward(mask, dest_idx)
        context.s, context.ec, context.h = s, ec, h

        return expert_input

    @staticmethod
    def backward(context, output_grad):
        mask, dest_idx = context.saved_tensors
        d_tokens = oslo_expert_parallel_cuda.dispatch_forward(
            context.s, context.ec, context.h, output_grad, mask, dest_idx
        )

        return d_tokens, None, None, None


class EPCombine(torch.autograd.Function):
    @staticmethod
    def forward(context, expert_tokens, logits, mask, dest_idx, ec):
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)

        fp16_flag = expert_tokens.dtype == torch.float16
        combine_inp = expert_tokens.to(torch.float32) if fp16_flag else expert_tokens
        ctokens = oslo_expert_parallel_cuda.combine_forward(
            s, e, c, h, combine_inp, logits, mask, dest_idx
        )
        output = ctokens.to(torch.float16) if fp16_flag else ctokens

        context.save_for_backward(expert_tokens, logits, mask, dest_idx)
        context.s, context.e, context.s, context.h = s, e, c, h
        context.fp16_flag = fp16_flag

        return output

    @staticmethod
    def backward(context, tokens_grad):
        expert_tokens, logits, mask, dest_idx = context.saved_tensors

        combine_grad = (
            tokens_grad.to(torch.float32)
            if tokens_grad.type is torch.float16
            else tokens_grad
        )

        combine_inp = (
            expert_tokens.to(torch.float32) if context.fp16_flag else expert_tokens
        )
        d_expert, d_logits = oslo_expert_parallel_cuda.combine_backward(
            context.s,
            context.e,
            context.s,
            context.h,
            combine_grad,
            combine_inp,
            logits,
            mask,
            dest_idx,
        )
        d_expert = d_expert.to(torch.float16) if context.fp16_flag else d_expert

        return d_expert, d_logits, None, None, None
