import os
from queue import Queue

import torch
import torch.distributed as dist
from torch.distributed import rpc

from oslo.torch.distributed import ParallelContext, ParallelMode


MessageQueues = [Queue() for _ in range(2)]


def rpc_push_queue(msg, ind):
    globals()["MessageQueues"][ind].put(msg)


def get_worker_map(parallel_context):
    world_size = parallel_context.get_world_size(
        parallel_mode=ParallelMode.PIPELINE,
    )
    return {i: f"rpc_{i:02d}" for i in range(world_size)}


def forward(tensor, shape, dtype, is_final_stage):
    result = tensor

    if is_final_stage:
        return shape, dtype
    else:
        return None


def test_rpc(parallel_context):
    try:
        torch.distributed.rpc.shutdown()
    except Exception:
        pass

    local_rank = parallel_context.get_local_rank(
        parallel_mode=ParallelMode.PIPELINE,
    )
    world_size = parallel_context.get_world_size(
        parallel_mode=ParallelMode.PIPELINE,
    )
    rpc.init_rpc(
        name=f"rpc_{local_rank:02d}",
        rank=local_rank,
        world_size=world_size,
        # backend
        # rpc_backend_options
    )

    worker_map = get_worker_map(parallel_context)
    is_first_stage = local_rank == 0
    is_final_stage = local_rank + 1 == world_size

    if is_first_stage:
        rpc.rpc_async(
            to=worker_map[local_rank + 1],
            func=rpc_push_queue,
            args=("No!!", 0),
        )

    elif is_final_stage:
        msg = MessageQueues[0].get()
        print(local_rank, msg)

    else:
        msg = MessageQueues[0].get()
        rpc.rpc_async(to=worker_map[local_rank + 1], func=rpc_push_queue, args=(msg, 0))


if __name__ == "__main__":
    parallel_context = ParallelContext.from_torch(
        data_parallel_size=1,
        pipeline_parallel_size=4,
        tensor_parallel_size=1,
    )

    test_rpc(parallel_context)
