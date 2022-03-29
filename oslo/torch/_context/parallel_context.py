from oslo._utils.logging import get_dist_logger


class ParallelContext(object):
    _instance = None

    @staticmethod
    def get_instance():
        if ParallelContext._instance is None:
            ParallelContext()
        return ParallelContext._instance

    def __init__(self):
        if self._instance is not None:
            raise Exception(
                "ParallelContext is a singleton class, "
                "You should get the instance by get_instance()."
            )
        else:
            ParallelContext._instance = self

        self._global_ranks = {}
        self._local_ranks = {}
        self._world_sizes = {}
        self._groups = {}
        self._ranks_in_group = {}

        self.world_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.tensor_parallel_size = 1
        self.virtual_pipeline_parallel_size = None
        self.virtual_pipeline_parallel_rank = None

        self._config = None
        self._verbose = False
        self._logger = get_dist_logger()
