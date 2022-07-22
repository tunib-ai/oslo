import time
from queue import Queue
import pickle
import concurrent.futures
from threading import Thread

import torch
import torch.distributed as dist
from torch.distributed import rpc

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._message import Message
from ._server import workers, _ORIGINAL_FORWARDS, _NEW_FORWARDS


MESSAGE_QUEUE_TYPES = ["FORWARD", "BACKWARD"]
MESSAGE_QUESES = {type: Queue() for type in MESSAGE_QUEUE_TYPES}


def rpc_push_queue(type: str, msg: Message):
    """
    Push message into queue

    Args:
        type (str): queue type
        msg (Message): message
    """
    globals()["MESSAGE_QUESES"][type].put(msg)


def rpc_pop_queue(type) -> Message:
    """
    Pop message from queue

    Args:
        type (str): queue type

    Returns:
        Message: message
    """
    return globals()["MESSAGE_QUESES"][type].get()


def receive(dtype, shape, src, dst):

    time.sleep(1)

    print('ong?')

    waiter = torch.tensor([1.]).to(dst)
    # dist.isend(starter, dst=src.index)
    torch.cuda.current_stream().synchronize()
    dist.send(waiter, dst=1)
    torch.cuda.current_stream().synchronize()

    buf = torch.zeros(shape, dtype=dtype, device=dst)

    print('ing?')

    dist.recv(tensor=buf, src=src.index)

    print(buf)


def pyobject_to_tensor(obj, fixed_buffer_size: int = 0) -> torch.Tensor:
    pickled = pickle.dumps(obj)
    result: torch.Tensor = torch.ByteTensor(bytearray(pickled))
    if fixed_buffer_size:
        delta = fixed_buffer_size - len(result)
        if delta < 0:
            raise ValueError(
                f"message too big to send, increase `fixed_buffer_size`? - {len(result)} > {fixed_buffer_size}"
            )
        elif delta > 0:
            result = torch.cat((result, torch.zeros(delta, dtype=torch.uint8)))

    return result


def tensor_to_pyobject(tensor: torch.Tensor):
    nparray = tensor.cpu().numpy()
    return pickle.loads(nparray.tobytes())


# TODO; can this be thread?
def prepare_recv(msg):
    msg = tensor_to_pyobject(msg)

    inputs = msg.inputs
    dtype = inputs.dtype
    shape = inputs.shape
    src = msg.src
    dst = msg.dst
    location = msg.location

    print(inputs)

    rank = dist.get_rank()
    torch.cuda.set_device(torch.distributed.get_rank())
    print(f'ing?? {torch.cuda.is_available()}, {dst}, {location}, {torch.cuda.current_device()}, {rank}')

    dummy = torch.rand(4, 4, 4)

    print(dummy)

    # dummy = dummy.to(
    #     f"cuda:{dst.index}"
    # )

    dummy = dummy.cuda()

    print(f"ang? {dummy.device}")

    with torch.cuda.device(dst):
        inputs = inputs.cuda()   # TODO; why this does not work?
        print(inputs.device)
        forward_fn = _ORIGINAL_FORWARDS[location]

    print(inputs)

    print(forward_fn)
    print(forward_fn(inputs))

    future = workers.put(forward_fn, inputs)

    result = future.result()
    result = pyobject_to_tensor(result)
    return result


class Communicator(object):
    """
    Communicator using Pytorch RPC module
    """

    def __init__(self, parallel_context: ParallelContext):
        self.rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)
