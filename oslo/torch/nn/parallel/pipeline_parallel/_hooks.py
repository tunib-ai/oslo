from queue import Queue
from types import MethodType
from collections.abc import Iterable

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import rpc
from torch.cuda.amp import custom_fwd, custom_bwd

from oslo.torch.distributed.nn.functional import send, recv
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.utils import get_parallel_context

from ._message import Message, TensorStub


# rpc queue
MessageQueue = Queue()


def rpc_push_queue(msg):
    globals()["MessageQueue"].put(msg)


def rpc_pull_queue():
    return MessageQueue.get()


PRE_FORWARD_HOOK_REGISTER = {}


def pipeline_parallel_pre_forward_hook(self, x):
    print('hou')

    if self.device == 'cpu':
        # module lives in other gpu
        dest = self.oslo_parallel[ParallelMode.PIPELINE]
        parallel_context = get_parallel_context(self, parallel_context=None)
        rpc_worker_name = parallel_context.get_pipeline_rpc_worker_name(dest)

        # rpc.rpc_async(
        #     to=rpc_worker_name,
        #     func=rpc_push_queue,
        #     args=("Ready?", ),
        # )

    else:
        return self(x)


def convert_forward(self):
    self.forward = MethodType(pipeline_parallel_pre_forward_hook, self)


def wrap_module(m, parallel_context):
    if isinstance(m, Iterable):     # ModuleList, ModuleDict...
        return m
    else:
        return ModuleWrapper(m, parallel_context)


class ModuleWrapper(nn.Module):
    def __init__(self, m, parallel_context):
        super().__init__()
        self.module = m
        self.parallel_context = parallel_context

    def forward(self, x):
        print(torch.cuda.current_device(), self.module._location)
        # print(x)

        self.send_message(x)

        # if torch.device('cuda', torch.cuda.current_device()) == x.device:
        #     x = self.module(x)  # TODO; need layer-wise split...
        #
        #     # Enqueue TODO; better way?
        #     self.send_message(x)
        # else:
        #     # module lives in other gpu
        #     self.send_message(x)

    def send_message(self, x):
        print("SEND")
        dst_rank = self.module.oslo_parallel[ParallelMode.PIPELINE]
        location = self.module._location
        rpc_worker_name = self.parallel_context.get_pipeline_rpc_worker_name(dst_rank)

        src_rank = dist.get_rank()
        msg = prepare_forward_message(x, src_rank, dst_rank, location)
        rpc.rpc_async(
            to=rpc_worker_name,
            func=rpc_push_queue,
            args=(msg, ),
        )


def prepare_event_head_message(x):
    return prepare_forward_message(x, 0, 0, "")


def prepare_forward_message(x, src, dst, location):
    msg = Message()
    # TODO; right???
    msg.comm_type = "request"   # TODO;
    msg.reqeust_from = "root"   # TODO;
    msg.exec_type = "forward"
    msg.inputs = None           # TODO; temp
    msg.outputs = None
    # msg.inputs = TensorStub()   # TODO; make with the information in x
    msg.src_rank = src
    msg.dst_rank = dst
    msg.location = location
    msg.in_autocast_context = False
    msg.in_grad_related_context = False
    msg.use_activation_checkpointing = False

    return msg
