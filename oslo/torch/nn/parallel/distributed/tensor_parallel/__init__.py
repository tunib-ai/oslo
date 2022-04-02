from typing import Optional

import torch.nn as nn
import torch.distributed as dist


class TensorParallel(nn.Module):
    """
    Tensor parallel module

    Args:
        module (nn.Module): PyTorch module object
        mode (str): Mode of tensor parallelism (1d, 2d, 2.5d, 3d)
        process_group (dist.ProcessGroup): process group object

    Notes:
        1. Similar design with torch.nn.parallel.DistributedDataParallel.
        2. Support auto de-parallelism

    Examples:
        >>> from oslo.torch.nn.parallel import TensorParallel

        >>> model = AnyTransformerModel()
        >>> optimizer = AnyOptimizer(model.paramters(), lr=3e-5)
        >>> tp_wrapper = TensorParallel(model, process_group=None, ...)

        >>> output = tp_wrapper(input_data)
        >>> output.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        mode: str,
        process_group=Optional[dist.ProcessGroup],
    ):
        super().__init__()
