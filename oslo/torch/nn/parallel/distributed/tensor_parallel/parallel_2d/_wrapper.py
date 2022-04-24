from oslo.torch.nn.parallel.utils import ParallelWrapper


class _TensorParallel2D(ParallelWrapper):
    """
    PyTorch module for 2D tensor parallelism

    Args:
        module (nn.Module): model object
        parallel_context (ParallelContext): parallel context object
    """

    def __init__(self, module, parallel_context):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context
