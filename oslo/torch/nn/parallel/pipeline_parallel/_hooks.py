import inspect
import time
import pickle
from contextlib import contextmanager
from queue import Queue
from types import MethodType
from collections.abc import Iterable
import random

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import rpc
from torch.cuda.amp import custom_fwd, custom_bwd

from oslo.torch.distributed.nn.functional import send, recv
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.utils import get_parallel_context
from oslo.torch.distributed.nn.functional import send, recv

from ._communicator import rpc_push_queue, prepare_recv, push_job_queue, LOCAL_RESULT_QUEUES
from ._server import workers, _ORIGINAL_FORWARDS, _NEW_FORWARDS, _MODULE_DEVICE_LOCATIONS
from ._message import HandShakeMessage, TensorStub


def on_same_device(location):
    module_device = _MODULE_DEVICE_LOCATIONS[location]
    module_device = torch.device('cuda', module_device)
    current_device = dist.get_rank()    # TODO;
    current_device = torch.device('cuda', current_device)
    is_same = module_device == current_device
    print(f'{location}, module device: {module_device}, current device: {current_device}, {is_same}')
    return is_same


# TODO; better way?
def get_current_location():
    frame = inspect.currentframe()
    while hasattr(frame, "f_back"):
        f_locals = frame.f_locals
        if "self" in f_locals and isinstance(f_locals["self"], nn.Module):
            break
        else:
            frame = frame.f_back

    caller_module = frame.f_locals["self"]
    location = caller_module.location

    return location


def wrap_forward(module):
    orig_forward = module.forward
    loc = module.location
    device = module.oslo_parallel[ParallelMode.PIPELINE]

    _ORIGINAL_FORWARDS[loc] = orig_forward
    _MODULE_DEVICE_LOCATIONS[loc] = device

    # TODO; make available on any input
    def new_forward(x, location=None):
        print(f'in new forward: {location=}')

        if location is None:
            location = get_current_location()

        is_same = on_same_device(location)

        print(f'{location}, {is_same} !!!!!!!!!!!!!!!!!!!')

        if is_same:
            # just put task in the job Queue
            forward_fn = _ORIGINAL_FORWARDS[location]

            print(x.device, dist.get_rank(), workers.rank)

            future = workers.put(forward_fn, x)
            result = future.result()

        else:
            src = dist.get_rank()
            src = torch.device('cuda', src)
            dst = _MODULE_DEVICE_LOCATIONS[location]
            dst = torch.device('cuda', dst)

            tag = random.randint(0, 12345678)   # TODO;
            LOCAL_RESULT_QUEUES[tag] = Queue()
            make_request(x, src, dst, location, tag)

            result = LOCAL_RESULT_QUEUES[tag].get()
            del LOCAL_RESULT_QUEUES[tag]

        return result

    module.forward = new_forward
    _NEW_FORWARDS[loc] = new_forward


def prepare_handshake_message(x, src, dst, location, tag):
    stub = TensorStub(
        id='0',   # TODO;
        dtype=x.dtype,
        shape=x.shape,
        requires_grad=x.requires_grad,
    )

    msg = HandShakeMessage(
        inputs=x.detach().clone().cpu(),
        src=src,
        dst=dst,
        location=location,
        id=tag,
    )

    return msg


def tensor_to_pyobject(tensor: torch.Tensor):
    nparray = tensor.cpu().numpy()
    return pickle.loads(nparray.tobytes())


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


def make_request(x, src, dst, location, tag):
    # TODO; this is super slow...
    msg = prepare_handshake_message(x, src, dst, location, tag)
    # msg = pyobject_to_tensor(msg)
    rpc_dst = f'PP_WORKER_{dst.index}'  # TODO;

    print(f'RPC dst: {rpc_dst}, {location=}')

    rpc.rpc_async(
        to=rpc_dst,
        func=push_job_queue,
        # args=((rpc.RRef(torch.rand(8, 8, 8)), msg), ),
        args=((x.cuda(), msg), ),
    )
