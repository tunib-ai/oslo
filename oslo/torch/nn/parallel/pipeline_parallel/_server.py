import queue
import mpi4py


class ExecutionServer(object):
    def __init__(self, module):
        self.module = module
        self.input_queue = queue.Queue()
        self.comm = mpi4py.MPI
