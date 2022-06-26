from threading import Thread

from oslo.torch.nn.parallel.pipeline_parallel._utils import WorkerThreadState


class CommInitiatorThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class D2DAllocatorThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class D2DSendThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class D2DRecvThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class CrossNodeProgressThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class IOSendCompletionWatcherThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class IORecvCompletionWatcherThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class CommRequesterThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class CommTrackerThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class CommPreprocessingThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class CommPostprocessingThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        pass


class WorkerThread(Thread):
    def __init__(self):
        super().__init__()
        self.state = WorkerThreadState.IDLE

    def run(self):
        pass
