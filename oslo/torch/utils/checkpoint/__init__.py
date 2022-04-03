from oslo.torch.utils.checkpoint._checkpoint_function import CheckpointFunction
from oslo.torch.utils.checkpoint._checkpoint_partitioner import CheckpointPartitioner
from oslo.torch.utils.checkpoint._rng_state_tracker import CudaRNGStatesTracker


RNG_TRACKER = None
PARTITIONER = None

DEFAULT_PARAMS = {
    "process_group": None,
    "num_layers": None,
    "partitioned_checkpointing": False,
    "cpu_checkpointing": False,
    "contiguous_checkpointing": False,
}


def checkpoint(function, *args, **kwargs):
    """
    Activation checkpoint function.

    Args:
        function: user function object
        *args: arguments of user function
        **kwargs: arguments of checkpoint function

    Returns:
        Tuple: output of user function
    """
    global RNG_TRACKER, PARTITIONER
    params = DEFAULT_PARAMS.copy()

    for key, val in kwargs.items():
        if key not in params:
            raise ValueError(
                f"{key} is unsupported keyword argument (kwargs), "
                f"only {list(params.keys())} are supported for kwargs of `checkpoint` function."
            )
        else:
            params[key] = val

    if params["partitioned_checkpointing"]:
        assert params["process_group"] is not None, (
            "If the param `partitioned_checkpointing` is True, `process_group` must not be None. "
            "Please input `process_group` like `checkpoint(fn, *args, process_group=YOUR_GROUP)`."
        )
        assert params["process_group"].size() > 1, (
            "If the param `partitioned_checkpointing` is True, "
            "the size of `process_group` must be greather than 1."
        )

    if params["contiguous_checkpointing"]:
        assert params["partitioned_checkpointing"] is True, (
            "`contiguous_checkpointing` can be used if `partitioned_checkpointing` is Ture. "
            "Please set `partitioned_checkpointing` to True "
            "like `checkpoint(fn, *args, partitioned_checkpointing=True, contiguous_checkpointing=True)."
        )
        assert params["num_layers"] is not None, (
            "If the param `contiguous_checkpointing` is True, `num_layers` must not be None. "
            "Please input the number of layer "
            "like `checkpoint(fn, *args, num_layers=YOUR_NUM_LAYER)."
        )

    if RNG_TRACKER is None:
        RNG_TRACKER = CudaRNGStatesTracker()

    if PARTITIONER is None:
        PARTITIONER = CheckpointPartitioner(
            process_group=params["process_group"],
            num_layers=params["num_layers"],
            partitioned_checkpointing=params["partitioned_checkpointing"],
            contiguous_checkpointing=params["contiguous_checkpointing"],
            cpu_checkpointing=params["cpu_checkpointing"],
        )

    all_outputs = []
    options = {"rng_tracker": RNG_TRACKER, "partitioner": PARTITIONER}
    CheckpointFunction.apply(function, options, all_outputs, *args)
    return tuple(all_outputs)
