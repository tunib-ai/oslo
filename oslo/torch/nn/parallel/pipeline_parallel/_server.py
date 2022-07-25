import concurrent.futures
from queue import Queue

import torch.distributed as dist


# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()

# Queue for rpc request
REMOTE_JOB_QUEUE = Queue()

# Queue for rpc response
LOCAL_RESULT_QUEUES = dict()

# Queue for final loss
FINAL_RESULT_QUEUE = Queue()


def push_final_result_queue(result):
    globals()["FINAL_RESULT_QUEUE"].put(result)


def push_result_queue(result, tag):
    globals()["LOCAL_RESULT_QUEUES"][tag].put(result)


def push_job_queue(req, *args, **kwargs):
    job = (req, args, kwargs)
    globals()["REMOTE_JOB_QUEUE"].put(job)


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
