from abc import ABC, abstractmethod


class ProcessGroupInitializer(ABC):
    """
    The abstract class for process group initialization.

    Args:
        rank (int): The rank of current process
        world_size (int): Size of whole communication world
        data_parallel_size (int): Size of data parallelization
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        data_parallel_size: int,
        sequence_parallel_size: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        expert_parallel_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.data_parallel_size = data_parallel_size
        self.sequence_parallel_size = sequence_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size

    @abstractmethod
    def init_dist_group(self):
        raise NotImplementedError
