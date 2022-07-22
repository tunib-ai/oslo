import concurrent.futures

import torch.distributed as dist


# original forward dictionary
_ORIGINAL_FORWARDS = dict()

# new forward dictionary
_NEW_FORWARDS = dict()

# module device locations
_MODULE_DEVICE_LOCATIONS = dict()


class Workers:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.executor._max_workers = 32   # TODO;
        self.rank = dist.get_rank()

    def put(self, forward_fn, *args):
        print(forward_fn)
        future = self.executor.submit(forward_fn, *args)
        return future


workers = Workers()


