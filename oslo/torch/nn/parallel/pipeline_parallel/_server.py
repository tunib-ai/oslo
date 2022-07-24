import os
import concurrent.futures

import torch.cuda
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
        self.rank = None

    def put(self, forward_fn, *args, **kwargs):
        # lazy initialization
        if self.rank is None:
            self.rank = dist.get_rank()
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {os.getpid()=}, Worker at {self.rank}'s current device: {torch.cuda.current_device()}")

        print(forward_fn)
        future = self.executor.submit(forward_fn, *args, **kwargs)
        return future


workers = Workers()
