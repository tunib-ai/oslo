from queue import Queue

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._message import Message

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


class Communicator(object):
    """
    Communicator using Pytorch RPC module
    """

    def __init__(self, parallel_context: ParallelContext):
        self.rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
        self.world_size = parallel_context.get_world_size(ParallelMode.PIPELINE)
