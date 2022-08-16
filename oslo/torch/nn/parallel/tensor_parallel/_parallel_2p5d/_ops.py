from typing import Any, Tuple, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from oslo.torch.distributed import ParallelMode, ParallelContext
from oslo.torch.distributed.nn.functional import all_reduce, reduce_scatter, all_gather


def add_bias_2p5d(
    input: Tensor,
    bias: Tensor,
    output_size_per_partition: int,
    tesseract_dim: int,
    row_rank: int,
    col_rank: int,
    dep_rank: int,
    skip_bias_add: bool,
    data_parallel_rank: int,
    pipeline_parallel_rank: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    parallel_context: ParallelContext,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _Add_Bias_2p5D.apply(
        input,
        bias,
        output_size_per_partition,
        tesseract_dim,
        row_rank,
        col_rank,
        dep_rank,
        skip_bias_add,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
        parallel_context,
        col_parallel_mode,
    )


def layernorm_2p5d(
    input: Tensor,
    E_x: Tensor,
    Var_x: Tensor,
    hidden_size: int,
    parallel_context: ParallelContext,
    row_parallel_mode: ParallelMode,
) -> Tensor:
    return _Layernorm2p5D.apply(
        input, E_x, Var_x, hidden_size, parallel_context, row_parallel_mode
    )


def all_gather_tensor_2p5d(
    inputs: Tensor,
    dim: int,
    parallel_context: ParallelContext,
    col_parallel_mode: ParallelMode,
) -> Tensor:
    return _AllGatherTensor2p5D.apply(inputs, dim, parallel_context, col_parallel_mode)


def gather_batch_2p5d(
    inputs: Tensor,
    dim: int = 0,
    parallel_context: Optional[ParallelContext] = None,
) -> Tensor:
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

    if world_size <= 1:
        return inputs

    return all_gather_tensor_2p5d(
        inputs,
        dim=dim,
        parallel_context=parallel_context,
        col_parallel_mode=ParallelMode.TENSOR_2P5D_COL,
    )


def reduce_by_batch_2p5d(
    inputs, reduce_mean: bool, parallel_context: ParallelContext
) -> Tensor:
    return _ReduceByBatch2p5D.apply(inputs, reduce_mean, parallel_context)


def reduce_tensor_2p5d(
    inputs: Tensor, parallel_context: ParallelContext, parallel_mode: ParallelMode
) -> Tensor:
    return _ReduceTensor2p5D.apply(inputs, parallel_context, parallel_mode)


def reduce_scatter_tensor_2p5d(
    inputs: Tensor,
    dim: int,
    parallel_context: ParallelContext,
    parallel_mode: ParallelMode,
) -> Tensor:
    dim_size = inputs.size(dim)
    world_size = parallel_context.get_world_size(parallel_mode)
    assert (
        dim_size % world_size == 0
    ), f"The batch size ({dim_size}) is not a multiple of 2.5D size * depth ({world_size})."

    return _ReduceScatterTensor2p5D.apply(inputs, dim, parallel_context, parallel_mode)


def split_batch_2p5d(
    inputs: Tensor, dim: int, parallel_context: ParallelContext
) -> Tensor:
    dim_size = inputs.size(dim)
    world_size = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)

    if world_size <= 1:
        return inputs

    assert (
        dim_size % world_size == 0
    ), f"The batch size ({dim_size}) is not a multiple of 2.5D size * depth ({world_size})."

    col_chunked = torch.chunk(
        inputs, parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL), dim=dim
    )[parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)].contiguous()
    return col_chunked


def get_current_device():
    r"""
    Get current device.
    """
    return torch.cuda.current_device()


class Matmul_AB_2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        tesseract_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        dep_rank: int,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-2], "Invalid shapes: A={}, B={} for AB.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        B_list = [torch.empty_like(B) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_a = (
            tesseract_dim * col_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_b = (
            row_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opa = [None] * 2
        opb = [None] * 2

        A_list[0].copy_(A)
        B_list[0].copy_(B)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(tesseract_dim):
            if i != tesseract_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(
                    A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True
                )
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(
                    B_list[1 - cur],
                    src=src_b + tesseract_dim,
                    group=col_group,
                    async_op=True,
                )

            if opa[cur] is not None:
                opa[cur].wait()
            if opb[cur] is not None:
                opb[cur].wait()

            torch.addmm(C, A_list[cur], B_list[cur], out=C)
            cur = 1 - cur
            src_a += 1
            src_b += tesseract_dim
        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
            ctx.parallel_context = parallel_context
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_2p5D.apply(
                output_grad,
                B,
                ctx.tesseract_dim,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = Matmul_ATB_2p5D.apply(
                A,
                output_grad,
                ctx.tesseract_dim,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Matmul_ABT_2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        tesseract_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        dep_rank: int,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], "Invalid shapes: A={}, B={} for ABT.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        B_list = [torch.empty_like(B) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_b = (
            row_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_c = (
            tesseract_dim * col_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opb = [None] * 2
        opr = [None] * 2

        B_list[0].copy_(B)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(tesseract_dim):
            if i != tesseract_dim - 1:
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(
                    B_list[1 - cur],
                    src=src_b + tesseract_dim,
                    group=col_group,
                    async_op=True,
                )

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == row_rank:
                    C.copy_(C_list[cur])

            if opb[cur] is not None:
                opb[cur].wait()

            torch.matmul(A, B_list[cur].transpose(0, 1), out=C_list[cur])
            opr[cur] = dist.reduce(
                C_list[cur], dst=src_c, group=row_group, async_op=True
            )
            cur = 1 - cur
            src_b += tesseract_dim
            src_c += 1

        for op in opr:
            op.wait()

        if tesseract_dim - 2 == row_rank:
            C.copy_(C_list[cur])
        if tesseract_dim - 1 == row_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
            ctx.parallel_context = parallel_context
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_AB_2p5D.apply(
                output_grad,
                B,
                ctx.tesseract_dim,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = Matmul_ATB_2p5D.apply(
                output_grad,
                A,
                ctx.tesseract_dim,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Matmul_ATB_2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        tesseract_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        dep_rank: int,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
    ):

        assert A.shape[-2] == B.shape[-2], "Invalid shapes: A={}, B={} for ATB.".format(
            A.shape, B.shape
        )

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = parallel_context.get_group(row_parallel_mode)
        col_group = parallel_context.get_group(col_parallel_mode)

        src_a = (
            tesseract_dim * col_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        src_c = (
            row_rank
            + tesseract_dim**2 * dep_rank
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )

        opa = [None] * 2
        opr = [None] * 2

        A_list[0].copy_(A)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        cur = 0

        for i in range(tesseract_dim):
            if i != tesseract_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(
                    A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True
                )

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == col_rank:
                    C.copy_(C_list[cur])

            if opa[cur] is not None:
                opa[cur].wait()

            torch.matmul(A_list[cur].transpose(0, 1), B, out=C_list[cur])
            opr[cur] = dist.reduce(
                C_list[cur], dst=src_c, group=col_group, async_op=True
            )
            cur = 1 - cur
            src_a += 1
            src_c += tesseract_dim

        for op in opr:
            op.wait()

        if tesseract_dim - 2 == col_rank:
            C.copy_(C_list[cur])
        if tesseract_dim - 1 == col_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
            ctx.parallel_context = parallel_context
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_2p5D.apply(
                B,
                output_grad,
                ctx.tesseract_dim,
                ctx.A_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
            B_grad = Matmul_AB_2p5D.apply(
                A,
                output_grad,
                ctx.tesseract_dim,
                ctx.B_shape,
                ctx.row_rank,
                ctx.col_rank,
                ctx.dep_rank,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size,
                ctx.parallel_context,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
            )
        return (
            A_grad,
            B_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _Add_Bias_2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        input: Tensor,
        bias: Tensor,
        output_size_per_partition: int,
        tesseract_dim: int,
        row_rank: int,
        col_rank: int,
        dep_rank: int,
        skip_bias_add: bool,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        parallel_context: ParallelContext,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        if col_rank == 0:
            bias_temp = bias.clone()
        else:
            bias_temp = torch.zeros(
                output_size_per_partition, dtype=bias.dtype, device=get_current_device()
            )
        src_rank = (
            row_rank
            + dep_rank * tesseract_dim**2
            + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
            + pipeline_parallel_rank * tensor_parallel_size
        )
        dist.broadcast(
            bias_temp, src=src_rank, group=parallel_context.get_group(col_parallel_mode)
        )

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.bias = skip_bias_add
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
            ctx.parallel_context = parallel_context
            ctx.col_parallel_mode = col_parallel_mode

        if skip_bias_add:
            return bias_temp
        else:
            output = input + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        tesseract_dim = ctx.tesseract_dim
        row_rank = ctx.row_rank
        col_rank = ctx.col_rank
        dep_rank = ctx.dep_rank
        data_parallel_rank = ctx.data_parallel_rank
        pipeline_parallel_rank = ctx.pipeline_parallel_rank
        pipeline_parallel_size = ctx.pipeline_parallel_size
        tensor_parallel_size = ctx.tensor_parallel_size
        parallel_context = ctx.parallel_context
        col_parallel_mode = ctx.col_parallel_mode

        if ctx.bias:
            dst_rank = (
                row_rank
                + dep_rank * (tesseract_dim**2)
                + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
                + pipeline_parallel_rank * tensor_parallel_size
            )
            dist.reduce(
                output_grad,
                dst=dst_rank,
                group=parallel_context.get_group(col_parallel_mode),
            )
            if col_rank == 0:
                return (
                    None,
                    output_grad,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            else:
                grad_tmp = torch.zeros_like(output_grad)
                return (
                    None,
                    grad_tmp,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            dst_rank = (
                row_rank
                + dep_rank * (tesseract_dim**2)
                + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size
                + pipeline_parallel_rank * tensor_parallel_size
            )
            dist.reduce(
                reduce,
                dst=dst_rank,
                group=parallel_context.get_group(col_parallel_mode),
            )
            if col_rank == 0:
                return (
                    output_grad,
                    reduce,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            else:
                reduce_tmp = torch.zeros_like(reduce)
                return (
                    output_grad,
                    reduce_tmp,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )


class _Layernorm2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        input: Tensor,
        E_x: Tensor,
        Var_x: Tensor,
        hidden_size: int,
        parallel_context: ParallelContext,
        row_parallel_mode: ParallelMode,
    ) -> Tensor:
        input = input - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        output = input * Var_x
        if ctx:
            ctx.hidden_size = hidden_size
            ctx.save_for_backward(output, Var_x)
            ctx.parallel_context = parallel_context
            ctx.row_parallel_mode = row_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        x, Var_x = ctx.saved_tensors
        row_parallel_mode = ctx.row_parallel_mode
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        with torch.no_grad():
            output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
            dist.all_reduce(
                output_grad_sum, group=ctx.parallel_context.get_group(row_parallel_mode)
            )
            output_grad_sum /= ctx.hidden_size

            output_grad_mul_x_sum = torch.sum(output_grad * x, dim=-1, keepdim=True)
            dist.all_reduce(
                output_grad_mul_x_sum,
                group=ctx.parallel_context.get_group(row_parallel_mode),
            )
            output_grad_mul_x_sum /= ctx.hidden_size

            input_grad = output_grad.clone()
            input_grad -= x * output_grad_mul_x_sum
            input_grad -= output_grad_sum
            input_grad *= Var_x

        return input_grad, None, None, None, None, None


class _AllGatherTensor2p5D(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        inputs: Tensor,
        dim: int,
        parallel_context: ParallelContext,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.dim = dim
        ctx.parallel_context = parallel_context
        ctx.col_parallel_mode = col_parallel_mode

        outputs = all_gather(
            inputs,
            dim,
            parallel_context=parallel_context,
            parallel_mode=col_parallel_mode,
        )

        if ctx:
            ctx.dim = dim
            ctx.parallel_context = parallel_context
            ctx.parallel_mode = col_parallel_mode

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad = reduce_scatter(
            output_grad,
            ctx.dim,
            parallel_context=ctx.parallel_context,
            parallel_mode=ctx.col_parallel_mode,
        )
        return grad.contiguous(), None, None, None


class SplitFirst(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        inputs: Tensor,
        tesseract_dim: int,
        parallel_context: ParallelContext,
        col_parallel_mode: ParallelMode,
    ) -> Tensor:
        col_rank = parallel_context.get_local_rank(col_parallel_mode)
        outputs = inputs.chunk(tesseract_dim, dim=0)[col_rank]

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.batch_size = inputs.size(0)
            ctx.parallel_context = parallel_context
            ctx.col_parallel_mode = col_parallel_mode
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad_shape = (ctx.batch_size,) + output_grad.shape[1:]
        grad = torch.empty(
            grad_shape, dtype=output_grad.dtype, device=get_current_device()
        )
        dist.all_gather(
            list(grad.chunk(ctx.tesseract_dim, dim=0)),
            output_grad.contiguous(),
            group=ctx.parallel_context.get_group(ctx.col_parallel_mode),
        )
        return grad, None, None, None


class _ReduceTensor2p5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode,
    ):
        return all_reduce(
            inputs,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor):
        return output_grad, None, None


class _ReduceScatterTensor2p5D(torch.autograd.Function):
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
    def backward(ctx, output_grad):
        return (
            all_gather(
                output_grad,
                ctx.dim,
                parallel_context=ctx.parallel_context,
                parallel_mode=ctx.parallel_mode,
            ),
            None,
            None,
            None,
        )


class _ReduceByBatch2p5D(torch.autograd.Function):
    @staticmethod
    def symbolic(
        graph,
        inputs,
        reduce_mean: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        output = all_reduce(
            inputs,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_COL,
        )
        if reduce_mean:
            reduce_size = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
            return output / reduce_size
        return output

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx: Any,
        inputs: Tensor,
        reduce_mean: bool = False,
        parallel_context: Optional[ParallelContext] = None,
    ):
        output = all_reduce(
            inputs,
            parallel_context=parallel_context,
            parallel_mode=ParallelMode.TENSOR_2P5D_COL,
        )
        if ctx:
            ctx.reduce_mean = reduce_mean

        if reduce_mean:
            reduce_size = parallel_context.get_world_size(ParallelMode.TENSOR_2P5D_COL)
            output /= reduce_size
            if ctx:
                ctx.reduce_size = reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor):
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None
        else:
            return output_grad, None, None


def split_batch_2d(parallel_context, tensor, tesseract_dim):
    tensor = tensor.chunk(tesseract_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    return tensor


def split_2p5d(parallel_context, tensor, tesseract_dim, col_first=True):
    if col_first:
        tensor = tensor.chunk(tesseract_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        ]
        tensor = tensor.chunk(tesseract_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        ]
    else:
        tensor = tensor.chunk(tesseract_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        ]
        tensor = tensor.chunk(tesseract_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        ]
    tensor = torch.chunk(tensor, tesseract_dim, dim=0)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_DEP)
    ]
    return tensor


def split_2d(parallel_context, tensor, tesseract_dim, col_first=True):
    if col_first:
        tensor = tensor.chunk(tesseract_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        ]
        tensor = tensor.chunk(tesseract_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        ]
    else:
        tensor = tensor.chunk(tesseract_dim, dim=-1)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_ROW)
        ]
        tensor = tensor.chunk(tesseract_dim, dim=0)[
            parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
        ]
    return tensor


def split_1d(parallel_context, tensor, summa_dim, dim=-1):
    tensor = tensor.chunk(summa_dim, dim=dim)[
        parallel_context.get_local_rank(ParallelMode.TENSOR_2P5D_COL)
    ]
    return tensor


def gather_2p5d(parallel_context, tensor, tesseract_dim, col_first=True):
    tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
    dist.all_gather(
        tensor_list, tensor, parallel_context.get_group(ParallelMode.TENSOR_2P5D_DEP)
    )
    tensor = torch.cat(tensor_list, dim=0)
    if col_first:
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor.contiguous(),
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL),
        )
        tensor = torch.cat(tensor_list, dim=0)
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor,
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
        )
        tensor = torch.cat(tensor_list, dim=-1)
    else:
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor,
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
        )
        tensor = torch.cat(tensor_list, dim=-1)
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor.contiguous(),
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL),
        )
        tensor = torch.cat(tensor_list, dim=0)
    return tensor


def gather_2d(parallel_context, tensor, tesseract_dim, col_first=True):
    if col_first:
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor.contiguous(),
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL),
        )
        tensor = torch.cat(tensor_list, dim=0)
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor,
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
        )
        tensor = torch.cat(tensor_list, dim=-1)
    else:
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor,
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_ROW),
        )
        tensor = torch.cat(tensor_list, dim=-1)
        tensor_list = [torch.zeros_like(tensor) for _ in range(tesseract_dim)]
        dist.all_gather(
            tensor_list,
            tensor.contiguous(),
            parallel_context.get_group(ParallelMode.TENSOR_2P5D_COL),
        )
        tensor = torch.cat(tensor_list, dim=0)
    return tensor


def gather_1d(parallel_context, tensor, summa_dim, dim=-1):
    parallel_modde = ParallelMode.TENSOR_2P5D_ROW
    tensor_list = [torch.zeros_like(tensor) for _ in range(summa_dim)]
    dist.all_gather(
        tensor_list,
        tensor.contiguous(),
        parallel_context.get_group(parallel_modde),
    )
    tensor = torch.cat(tensor_list, dim=dim)
    return tensor
