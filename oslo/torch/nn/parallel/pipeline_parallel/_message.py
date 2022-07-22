from dataclasses import dataclass
from typing import Optional, Any
from typing import Tuple, List, Union

import torch

MESSAGE_GENERATION = 0


@dataclass(init=False)
class Message:
    comm_type: str
    # 1. request or response
    reqeust_from: Optional[str]
    # 2. reqeust module id
    exec_type: str
    # 3. forward or backward
    inputs: Optional[Any]
    # 4. input data for module execution
    outputs: Optional[Any]
    # 5. output data from module execution
    src_rank: int
    # 6. source pp rank
    dst_rank: int
    # 7. destination pp rank
    location: int
    # 8. The location of the module within the module graph
    in_autocast_context: bool
    # 9. Whether the requester is currently in a autocast context
    in_grad_related_context: bool
    # 10. Whether the requester is currently in a no grad/enable grad context
    use_activation_checkpointing: bool
    # 11. Whether activation checkpointing is enabled for the current module

    def __init__(self):
        global MESSAGE_GENERATION
        MESSAGE_GENERATION += 1
        self.tag = MESSAGE_GENERATION


@dataclass
class TensorStub(object):
    id: str
    dtype: torch.dtype
    shape: Union[List, Tuple]
    requires_grad: bool


# TODO; dummy object for test
@dataclass
class HandShakeMessage:
    inputs: TensorStub
    src: torch.device
    dst: torch.device
    location: str
