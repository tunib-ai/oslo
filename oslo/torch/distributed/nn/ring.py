import torch
import torch.distributed as dist
from torch.cuda.amp import custom_bwd, custom_fwd

from oslo.torch.distributed import ParallelContext, ParallelMode


def send_forward_recv_forward(
    tensor_send_next: torch.Tensor,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
):
    """
    Sends a tensor to the next member and receives a tensor from the previous member.
    This function returns the received tensor from the previous member.
    This function assumes that the receiving tensor has the same shape with the sending tensor.

    This function is based on the `send_forward_recv_forward` implementation of Megatron-LM.
    Args:
        tensor_send_next: Tensor sent to next member
        parallel_context: ParallelContext holding process group information
        parallel_mode: Parallel group mode used in this communication
    Returns:
        :class:`torch.Tensor`: The tensor received from the previous.
    """

    buffer_shape = tensor_send_next.size()
    tensor_recv_prev = torch.empty(
        buffer_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=tensor_send_next.dtype,
    )

    ops = []
    # send to next rank
    send_next_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_send_next,
        parallel_context.get_next_global_rank(parallel_mode),
    )
    ops.append(send_next_op)

    # receive from prev rank
    recv_prev_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv_prev,
        parallel_context.get_prev_global_rank(parallel_mode),
    )
    ops.append(recv_prev_op)

    # TODO; need this ? ColossalAI implementation uses this
    # current_rank = parallel_context.get_global_rank()
    # if current_rank % 2 == 0:
    #     ops = ops[::-1]

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()  # TODO; need to check whether cuda is available or not?

    return tensor_recv_prev


class _RingQK(torch.autograd.Function):
    """
    Calculate QK^T in a ring-exchange style
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx, sub_q: torch.Tensor, sub_k: torch.Tensor, parallel_context: ParallelContext
    ):
        # save tensor for backward
        ctx.save_for_backward(sub_q, sub_k)
        # save process group information
        # need to save for send_forward_recv_forward in backward pass
        ctx.parallel_context = parallel_context

        bsz, len_sub_q, d = sub_q.shape
        assert bsz == sub_k.size(0), "Q and K must have same embedding size."
        assert d == sub_k.size(-1), "Q and K must have same embedding size."
        len_sub_k = sub_k.size(1)

        # get process group information and calculate total length of key
        local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)
        len_k = len_sub_k * local_world_size

        # create local segment of attention score
        sub_attn = torch.empty(
            bsz,
            len_sub_q,
            len_k,  # shape of sub-attention matrix
            device=torch.cuda.current_device(),
            dtype=sub_q.dtype,
        )

        # compute local QK^T
        sub_attn_part = torch.einsum("b q d, b k d -> b q k", sub_q, sub_k)
        start_idx = local_rank * len_sub_k
        end_idx = (local_rank + 1) * len_sub_k
        sub_attn[:, :, start_idx:end_idx] = sub_attn_part

        # to send/recv in proper order
        sub_k = sub_k.contiguous()
        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = send_forward_recv_forward(
                sub_k, parallel_context, ParallelMode.SEQUENCE
            )
            sub_attn_part = torch.einsum("b q d, b k d -> b q k", sub_q, sub_k)

            sender_local_rank = (local_rank - 1 - i) % local_world_size
            start_idx = sender_local_rank * len_sub_k
            end_idx = (sender_local_rank + 1) * len_sub_k
            sub_attn[:, :, start_idx:end_idx] = sub_attn_part

        return sub_attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (
            sub_q,
            sub_k,
        ) = ctx.saved_tensors
        parallel_context = ctx.parallel_context

        local_rank = parallel_context.local_rank
        local_world_size = parallel_context.local_world_size
        len_sub_k = sub_k.size(1)

        # calculate local gradient of sub_k
        grad_k = torch.einsum("b q k, b q d -> b k d", grad_output, sub_q)
        # accumulate gradients
        dist.all_reduce(grad_k, group=parallel_context.get_group(ParallelMode.SEQUENCE))
        # slice for sub-sequence
        start_idx = local_rank * len_sub_k
        end_idx = (local_rank + 1) * len_sub_k
        grad_k = grad_k[:, start_idx:end_idx]

        # calculate gradient for sub_q
        grad_q = torch.zeros_like(
            sub_q,
            device=torch.cuda.current_device(),
            dtype=sub_q.dtype,
        )

        # compute with local sub_k
        start_idx = local_rank * len_sub_k
        end_idx = (local_rank + 1) * len_sub_k
        grad_q += torch.einsum(
            "b q k, b k d -> b q d", grad_output[:, :, start_idx:end_idx], sub_k
        )

        # to send/recv in proper order
        sub_k = sub_k.contiguous()
        # compute (dL/dZ)K in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = send_forward_recv_forward(
                sub_k, parallel_context, ParallelMode.SEQUENCE
            )
            sender_local_rank = (local_rank - 1 - i) % local_world_size
            start_idx = sender_local_rank * len_sub_k
            end_idx = (sender_local_rank + 1) * len_sub_k
            grad_q += torch.einsum(
                "b q k, b k d -> b q d", grad_output[:, :, start_idx:end_idx], sub_k
            )

        return grad_q, grad_k, None, None, None


class _RingAV(torch.autograd.Function):
    """
    Calculate AV in a ring-exchange style
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, sub_attn, sub_v, parallel_context: ParallelContext):
        # save tensor for backward
        ctx.save_for_backward(sub_attn, sub_v)
        # save process group information
        # need to save for send_forward_recv_forward in backward pass
        ctx.parallel_context = parallel_context

        bsz, len_sub_v, d = sub_v.shape
        len_sub_attn = sub_attn.size(1)

        # get process group information
        local_rank = parallel_context.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = parallel_context.get_world_size(ParallelMode.SEQUENCE)

        # create local segment of output
        sub_output = torch.zeros(
            bsz,
            len_sub_attn,
            d,
            device=torch.cuda.current_device(),
            dtype=sub_attn.dtype,
        )

        # compute local AV
        start_idx = local_rank * len_sub_v
        end_idx = (local_rank + 1) * len_sub_v
        sub_output += torch.einsum(
            "b q k, b k d -> b q d", sub_attn[:, :, start_idx:end_idx], sub_v
        )

        # to send/recv in proper order
        sub_v = sub_v.contiguous()
        # compute AV in ring - all - reduce style
        for i in range(local_world_size - 1):
            sub_v = send_forward_recv_forward(
                sub_v, parallel_context, ParallelMode.SEQUENCE
            )
            sender_local_rank = (local_rank - 1 - i) % local_world_size
            start_idx = sender_local_rank * len_sub_v
            end_idx = (sender_local_rank + 1) * len_sub_v
            # kevin: I changed `len_sub_k` -> `len_sub_v` for pre-commit.
            # If you checked this, please remove this comments :)
            sub_output += torch.einsum(
                "b q k, b k d -> b q d", sub_attn[:, :, start_idx:end_idx], sub_v
            )

        return sub_output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        sub_attn, sub_v = ctx.saved_tensors
        parallel_context = ctx.parallel_context

        local_rank = parallel_context.local_rank
        local_world_size = parallel_context.local_world_size
        len_sub_v = sub_v.size(1)

        # calculate local gradient of v
        grad_v = torch.einsum("b q k, b q d -> b k d", sub_attn, grad_output)
        # accumulate gradients
        dist.all_reduce(grad_v, group=parallel_context.get_group(ParallelMode.SEQUENCE))
        # slice for sub-sequence
        start_idx = local_rank * len_sub_v
        end_idx = (local_rank + 1) * len_sub_v
        grad_v = grad_v[:, start_idx:end_idx]

        # calculate gradient for attention score
        grad_attn = torch.zeros_like(
            sub_attn,
            device=torch.cuda.current_device(),
            dtype=grad_output.dtype,
        )

        # compute with local sub_v
        grad_attn[:, :, start_idx:end_idx] += torch.einsum(
            "b q d, b k d -> b q k", grad_output, sub_v
        )

        # to send/recv in proper order
        sub_v = sub_v.contiguous()
        # compute (dL/dZ)V^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = send_forward_recv_forward(
                sub_v, parallel_context, ParallelMode.SEQUENCE
            )
            sender_local_rank = (local_rank - 1 - i) % local_world_size
            start_idx = sender_local_rank * len_sub_v
            end_idx = (sender_local_rank + 1) * len_sub_v
            grad_attn[:, :, start_idx:end_idx] += torch.einsum(
                "b q d, b k d -> b q k", grad_output, sub_v
            )

        return grad_attn, grad_v, None, None, None, None
