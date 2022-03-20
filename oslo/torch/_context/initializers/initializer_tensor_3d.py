from oslo.torch._context.initializers.initializer import (
    ProcessGroupInitializer,
)


class _TensorParallel3DInputGroupInitializer(ProcessGroupInitializer):
    """
    Process group initializer for input of 3D tensor parallelism.

    Args:
        num_group (int): The number of all tensor groups
        depth (int): Depth of 3D tensor parallelism
    """

    def __init__(self, num_group: int, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_group = num_group
        self.depth = depth

    def init_dist_group(self):
        pass
