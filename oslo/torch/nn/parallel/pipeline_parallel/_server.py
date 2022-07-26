import concurrent.futures
import time
from queue import Queue

import torch
import torch.distributed as dist


# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()

# Queue for rpc request
REMOTE_JOB_QUEUE = Queue()

# Queues for rpc response
REMOTE_RESULT_QUEUES = dict()

# Queue for final loss
FINAL_RESULT_QUEUE = Queue()

# Dict for activation
ACTIVATIONS = dict()


def push_final_result_queue(result):
    globals()["FINAL_RESULT_QUEUE"].put(result)


def push_result_queue(req, result, tag):
    result = (req, result)
    globals()["REMOTE_RESULT_QUEUES"][tag].put(result)


def push_job_queue(req, *args, **kwargs):
    job = (req, args, kwargs)
    globals()["REMOTE_JOB_QUEUE"].put(job)


def run_request_backward(tag, to, *grad_outputs):
    activation = ACTIVATIONS[tag]    # TODO;

    print(f'run_remote_backward, {dist.get_rank()=}, {to=}, {tag=}, {ACTIVATIONS.keys()=}')

    print(f'{activation=}')

    torch.autograd.backward(activation, grad_outputs)

    # del ACTIVATIONS[tag]


def run_response_backward(tag, to, *grad_outputs):
    activation = ACTIVATIONS[tag]    # TODO;

    print(f'run_remote_backward, {dist.get_rank()=}, {to=}, {tag=}, {ACTIVATIONS.keys()=}')

    print(f'{activation=}')

    torch.autograd.backward(activation, grad_outputs)

    # del ACTIVATIONS[tag]


class Workers:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.rank = None

    def put(self, forward_fn, *args, **kwargs):
        # lazy initialization
        if self.rank is None:
            self.rank = dist.get_rank()

        future = self.executor.submit(forward_fn, *args, **kwargs)
        return future


workers = Workers()
