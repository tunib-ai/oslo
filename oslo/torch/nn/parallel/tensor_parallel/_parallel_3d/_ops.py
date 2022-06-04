from lib2to3.pgen2.token import OP
from typing import Any, Tuple, Optional

import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from oslo.torch.distributed import ParallelContext, ParallelMode, parallel_context
from oslo.torch.distributed.nn.functional import (
    all_gather,
    all_reduce,
    reduce_scatter,
    broadcast,
    reduce,
)


class _Linear3D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        inputs: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        input_dim: int = 0,
        weight_dim: int = -1,
        output_dim: int = 0,
        parallel_context: Optional[ParallelContext] = None,
        input_parallel_mode: Optional[ParallelMode] = None,
        weight_parallel_mode: Optional[ParallelMode] = None,
        output_parallel_mode: Optional[ParallelMode] = None,
    ) -> Tensor:

        inputs = all_gather(
            inputs,
            input_dim,
            parallel_context=parallel_context,
            parallel_mode=input_parallel_mode,
        )
        weight = all_gather(
            weight,
            weight_dim,
            parallel_context=parallel_context,
            parallel_mode=weight_parallel_mode,
        )

        output = torch.matmul(inputs, weight.t())
        output = reduce_scatter(
            output,
            output_dim,
            parallel_context=parallel_context,
            parallel_mode=output_parallel_mode,
        )

        if bias is not None:
            output += bias

        if ctx:
            ctx.save_for_backward(inputs, weight)
            ctx.use_bias = bias is not None
            ctx.input_dim = input_dim
            ctx.weight_dim = weight_dim
            ctx.output_dim = output_dim
            ctx.parallel_context = parallel_context
            ctx.input_parallel_mode = input_parallel_mode
            ctx.weight_parallel_mode = weight_parallel_mode
            ctx.output_parallel_mode = output_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        inputs, weight = ctx.saved_tensors
        with torch.no_grad():
            output_grad = all_gather(
                output_grad,
                ctx.output_dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.output_parallel_mode,
            )

            async_ops = list()

            input_grad = torch.matmul(output_grad, weight)
            input_grad, op = reduce_scatter(
                input_grad,
                ctx.input_dim,
                async_op=True,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.input_parallel_mode,
            )
            async_ops.append(op)

            weight_grad = torch.matmul(
                output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1),
                inputs.reshape(-1, inputs.shape[-1]),
            )
            weight_grad, op = reduce_scatter(
                weight_grad,
                ctx.weight_dim,
                async_op=True,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.weight_parallel_mode,
            )
            async_ops.append(op)

            if ctx.use_bias:
                bias_grad = torch.sum(
                    output_grad, dim=tuple(range(len(output_grad.shape))[:-1])
                )
                bias_grad, op = all_reduce(
                    bias_grad,
                    async_op=True,
                    parallel_context=ctx.parallel_context,
                    parallel_mode=ctx.weight_parallel_mode,
                )
                async_ops.append(op)
            else:
                bias_grad = None

            for op in async_ops:
                if op is not None:
                    op.wait()

        return (
            input_grad,
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def linear_3d(
    inputs: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_dim: int = 0,
    weight_dim: int = -1,
    output_dim: int = 0,
    parallel_context: Optional[ParallelContext] = None,
    input_parallel_mode: Optional[ParallelMode] = None,
    weight_parallel_mode: Optional[ParallelMode] = None,
    output_parallel_mode: Optional[ParallelMode] = None,
) -> Tensor:
    return _Linear3D.apply(
        inputs,
        weight,
        bias,
        input_dim,
        weight_dim,
        output_dim,
        parallel_context,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


class _Classifier3D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        parallel_context: ParallelContext,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:

        ranks_in_group = parallel_context.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[parallel_context.get_local_rank(output_parallel_mode)]
        weight = broadcast(weight, src_rank, parallel_context, input_parallel_mode)

        output = torch.matmul(inputs, weight.t())
        output = all_reduce(output, output_parallel_mode)

        if bias is not None:
            output += bias

        if ctx:
            ctx.save_for_backward(inputs, weight)
            ctx.use_bias = bias is not None
            ctx.src_rank = src_rank
            ctx.parallel_context = parallel_context
            ctx.input_parallel_mode = input_parallel_mode
            ctx.weight_parallel_mode = weight_parallel_mode
            ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        inputs, weight = ctx.saved_tensors
        with torch.no_grad():
            async_ops = list()

            weight_grad = torch.matmul(
                output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1),
                inputs.reshape(-1, inputs.shape[-1]),
            )
            weight_grad = reduce(
                weight_grad,
                ctx.src_rank,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.input_parallel_mode,
            )
            if ctx.parallel_context.get_local_rank(
                ctx.input_parallel_mode
            ) == ctx.parallel_context.get_local_rank(ctx.output_parallel_mode):
                weight_grad, op = all_reduce(
                    weight_grad,
                    ctx.parallel_context,
                    ctx.weight_parallel_mode,
                    async_op=True,
                )
                async_ops.append(op)
            else:
                weight_grad = None

            if ctx.use_bias:
                bias_grad = torch.sum(
                    output_grad, dim=tuple(range(len(output_grad.shape))[:-1])
                )
                bias_grad = all_reduce(
                    bias_grad,
                    parallel_context=ctx.parallel_context,
                    parallel_mode=ctx.input_parallel_mode,
                )
                bias_grad, op = all_reduce(
                    bias_grad,
                    async_op=True,
                    parallel_context=ctx.parallel_context,
                    parallel_mode=ctx.input_parallel_mode,
                )
                async_ops.append(op)
            else:
                bias_grad = None

            input_grad = torch.matmul(output_grad, weight)

            for op in async_ops:
                if op is not None:
                    op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None


def classifier_3d(
    inputs: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    parallel_context: ParallelContext,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D parallel classifier.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Classifier3D.apply(
        inputs,
        weight,
        bias,
        parallel_context,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


class _Layernorm3D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        normalized_shape: int,
        eps: float,
        parallel_context: ParallelContext,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        mean = (
            all_reduce(
                torch.sum(inputs, dim=-1, keepdim=True),
                parallel_context=parallel_context,
                parallel_mode=output_parallel_mode,
            )
            / normalized_shape
        )

        mu = inputs - mean

        var = (
            all_reduce(
                torch.sum(mu**2, dim=-1, keepdim=True),
                parallel_context=parallel_context,
                parallel_mode=output_parallel_mode,
            )
            / normalized_shape
        )

        sigma = torch.sqrt(var + eps)

        z = mu / sigma
        output = weight * z
        if bias is not None:
            output = output + bias

        if ctx:
            ctx.save_for_backward(mu, sigma, weight)
            ctx.use_bias = bias is not None
            ctx.normalized_shape = normalized_shape
            ctx.parallel_context = parallel_context
            ctx.input_parallel_mode = input_parallel_mode
            ctx.weight_parallel_mode = weight_parallel_mode
            ctx.output_parallel_mode = output_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        mu, sigma, weight = ctx.saved_tensors
        with torch.no_grad():
            weight_grad = output_grad * mu / sigma
            if ctx.use_bias:
                bias_grad = output_grad
                weight_grad = torch.stack([bias_grad, weight_grad]).contiguous()
            else:
                bias_grad = None
            weight_grad = torch.sum(
                weight_grad, dim=tuple(range(len(weight_grad.shape))[1:-1])
            )
            weight_grad = all_reduce(
                weight_grad,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.weight_parallel_mode,
            )
            weight_grad = all_reduce(weight_grad, ctx.input_parallel_mode)
            if ctx.use_bias:
                bias_grad, weight_grad = weight_grad[0], weight_grad[1]

            dz = output_grad * weight
            dvar = dz * mu * (-0.5) * sigma ** (-3)
            dvar = all_reduce(
                torch.sum(dvar, dim=-1, keepdim=True),
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.output_parallel_mode,
            )
            dmean = dz * (-1 / sigma) + dvar * -2 * mu / ctx.normalized_shape
            dmean = all_reduce(
                torch.sum(dmean, dim=-1, keepdim=True),
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.output_parallel_mode,
            )

            input_grad = (
                dz / sigma
                + dvar * 2 * mu / ctx.normalized_shape
                + dmean / ctx.normalized_shape
            )

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def layernorm_3d(
    inputs: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    normalized_shape: int,
    eps: float,
    parallel_context: ParallelContext,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D parallel Layernorm.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Layernorm3D.apply(
        inputs,
        weight,
        bias,
        normalized_shape,
        eps,
        parallel_context,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


def split_tensor_3d(
    tensor: Tensor,
    dim: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
) -> Tensor:
    r"""Splits 3D parallel tensor in specified dimension.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): Parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    dim_size = tensor.size(dim)
    world_size = parallel_context.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, (
        f"The dimension {dim} to split, size ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )
    if tensor.size(dim) <= 1:
        return tensor
    output = torch.chunk(
        tensor, parallel_context.get_world_size(parallel_mode), dim=dim
    )[parallel_context.get_local_rank(parallel_mode)].contiguous()
    return output


def split_batch_3d(
    inputs: Tensor,
    dim: int = 0,
    parallel_context: Optional[ParallelContext] = None,
    input_parallel_mode: ParallelMode = ParallelMode.TENSOR_3D_INPUT,
    weight_parallel_mode: ParallelMode = ParallelMode.TENSOR_3D_WEIGHT,
) -> Tensor:
    r"""Splits 3D tensor in batch.

    Args:
        input_ (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): weight parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    dim_size = inputs.size(dim)
    weight_world_size = parallel_context.get_world_size(weight_parallel_mode)
    input_world_size = parallel_context.get_world_size(input_parallel_mode)

    assert (
        dim_size % (input_world_size * weight_world_size) == 0
    ), f"The batch size ({dim_size}) is not a multiple of square of 3D depth ({input_world_size*weight_world_size})."

    if inputs.size(dim) <= 1:
        return inputs
    output = torch.chunk(inputs, weight_world_size, dim=dim)[
        parallel_context.get_local_rank(weight_parallel_mode)
    ].contiguous()
    output = torch.chunk(output, input_world_size, dim=dim)[
        parallel_context.get_local_rank(input_parallel_mode)
    ].contiguous()
    return output


class _ReduceTensor3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        return all_reduce(
            inputs, parallel_context=parallel_context, parallel_mode=parallel_mode
        )

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor):
        return output_grad, None, None


def reduce_tensor_3d(
    tensor: Tensor, parallel_context: ParallelContext, parallel_mode: ParallelMode
) -> Tensor:
    r"""All-reduce the input

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    return _ReduceTensor3D.apply(tensor, parallel_context, parallel_mode)


class _AllGatherTensor3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        dim: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
            ctx.parallel_mode = parallel_mode
        output = all_gather(inputs, dim, parallel_mode)
        return output

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor):
        input_grad = reduce_scatter(
            output_grad,
            ctx.dim,
            parallel_context=ctx.parallel_context,
            parallel_mode=ctx.parallel_mode,
        )
        return input_grad, None, None, None


def all_gather_tensor_3d(
    tensor: Tensor,
    dim: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
) -> Tensor:
    r"""All-reduce the gradient in backward pass.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    return _AllGatherTensor3D.apply(tensor, dim, parallel_context, parallel_mode)


class _ReduceScatterTensor3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        dim: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
            ctx.parallel_mode = parallel_mode

        return reduce_scatter(
            inputs,
            dim,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor):
        input_grad = all_gather(
            output_grad,
            ctx.dim,
            parallel_context=ctx.parallel_context,
            parallel_mode=ctx.parallel_mode,
        )
        return input_grad, None, None, None


def reduce_scatter_tensor_3d(
    tensor: Tensor,
    dim: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
) -> Tensor:
    r"""Reduce-scatter the input.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to scatter.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    dim_size = tensor.size(dim)
    world_size = parallel_context.get_world_size(parallel_mode)
    assert (
        dim_size % world_size == 0
    ), f"The batch size ({dim_size}) is not a multiple of square of 3D depth ({world_size})."

    return _ReduceScatterTensor3D.apply(tensor, dim, parallel_context, parallel_mode)


class _ReduceByBatch3D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        inputs: Tensor,
        reduce_mean: bool = False,
        parallel_context: Optional[ParallelContext] = None,
        input_parallel_mode: Optional[ParallelMode] = None,
        weight_parallel_mode: Optional[ParallelMode] = None,
    ) -> Tensor:
        output = all_reduce(inputs, input_parallel_mode)
        output = all_reduce(output, weight_parallel_mode)

        if ctx:
            ctx.reduce_mean = reduce_mean

        if reduce_mean:
            reduce_size = parallel_context.get_world_size(
                input_parallel_mode
            ) * parallel_context.get_world_size(weight_parallel_mode)
            output /= reduce_size
            if ctx:
                ctx.reduce_size = reduce_size

        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None, None, None
        else:
            return output_grad, None, None, None, None


def reduce_by_batch_3d(
    tensor: Tensor,
    reduce_mean: bool = False,
    parallel_context: Optional[ParallelContext] = None,
    input_parallel_mode: Optional[ParallelMode] = None,
    weight_parallel_mode: Optional[ParallelMode] = None,
) -> Tensor:
    r"""All-reduce the input from the model parallel region.

    Args:
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        reduce_mean (bool, optional): If set to ``True``, it will divide the output by
            (input parallel size * weight parallel size), default to False.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _ReduceByBatch3D.apply(
        tensor, reduce_mean, parallel_context, input_parallel_mode, weight_parallel_mode
    )


class _BroadcastWeight3D_FromDiagonal(torch.autograd.Function):
    r"""broadcast weight from diagonal.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        inputs: Tensor,
        parallel_context: ParallelContext,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        ranks_in_group = parallel_context.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[parallel_context.get_local_rank(output_parallel_mode)]
        output = broadcast(
            inputs,
            src_rank,
            parallel_context=parallel_context,
            parallel_mode=input_parallel_mode,
        )
        if ctx:
            ctx.src_rank = src_rank
            ctx.parallel_context = parallel_context
            ctx.input_parallel_mode = input_parallel_mode
            ctx.weight_parallel_mode = weight_parallel_mode
            ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_grad = reduce(
            output_grad,
            ctx.src_rank,
            parallel_context=ctx.parallel_context,
            parallel_mode=ctx.input_parallel_mode,
        )
        if parallel_context.get_local_rank(
            ctx.input_parallel_mode
        ) == parallel_context.get_local_rank(ctx.output_parallel_mode):
            input_grad = all_reduce(
                input_grad,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.weight_parallel_mode,
            )
        else:
            input_grad = None
        return input_grad, None, None, None, None


def broadcast_weight_3d_from_diagonal(
    tensor: Tensor,
    parallel_context: ParallelContext,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    return _BroadcastWeight3D_FromDiagonal.apply(
        tensor,
        parallel_context,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )
