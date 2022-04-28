from oslo.torch.nn.parallel.utils import (
    is_huggingface_model, 
    is_oslo_model, 
    ParallelWrapper, 
    _update_module_arguments,
)


class _TensorParallel2D(ParallelWrapper):
    """
    PyTorch module for 2D tensor parallelism

    Args:
        module (nn.Module): model object
        parallel_context (ParallelContext): parallel context object
    """

    def __init__(
        self, 
        module: nn.Module, 
        parallel_context: ParallelContext,
    ):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context
