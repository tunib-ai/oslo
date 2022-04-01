import functools
from contextlib import contextmanager

import torch.cuda
from torch import Tensor

from oslo.torch.distributed import ParallelMode
from oslo.torch.distributed._random.seed_manager import SeedManager

_SEED_MANAGER = SeedManager()


def get_seeds():
    return _SEED_MANAGER.seeds


def get_states(copy=False):
    states = _SEED_MANAGER.seed_states

    if copy:
        new_states = dict()

        for parallel_mode, state in states.items():
            new_states[parallel_mode] = state.clone()
        return new_states
    else:
        return _SEED_MANAGER.seed_states


def get_current_mode():
    return _SEED_MANAGER.current_mode


def add_seed(parallel_mode: ParallelMode, seed: int, overwrite: bool = False):
    _SEED_MANAGER.add_seed(parallel_mode, seed, overwrite)


def set_mode(parallel_mode: ParallelMode):
    _SEED_MANAGER.set_mode(parallel_mode)


def set_seed_states(parallel_mode: ParallelMode, state: Tensor):
    _SEED_MANAGER.set_state(parallel_mode, state)


def sync_states():
    current_mode = get_current_mode()
    current_states = torch.cuda.get_rng_state()
    set_seed_states(current_mode, current_states)


@contextmanager
def seed(parallel_mode: ParallelMode):
    try:
        # set to new mode
        current_mode = _SEED_MANAGER.current_mode
        yield _SEED_MANAGER.set_mode(parallel_mode)
    finally:
        # recover
        _SEED_MANAGER.set_mode(current_mode)


def with_seed(func, parallel_mode: ParallelMode):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # switch mode
        current_mode = _SEED_MANAGER.current_mode
        _SEED_MANAGER.set_mode(parallel_mode)

        # exec func
        out = func(*args, **kwargs)

        # recover state
        _SEED_MANAGER.set_mode(current_mode)

        return out

    return wrapper


def moe_set_seed(seed):
    if torch.cuda.is_available():
        from oslo.torch.distributed import ParallelContext

        global_rank = ParallelContext.get_instance().get_global_rank()
        diff_seed = seed + global_rank
        add_seed(ParallelMode.TENSOR, diff_seed, True)
        print(
            f"moe seed condition: {global_rank} with tensor seed {diff_seed}",
            flush=True,
        )


def reset_seeds():
    _SEED_MANAGER.reset()
