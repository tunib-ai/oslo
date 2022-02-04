import contextlib
import copy

import torch
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager


class CudaRNGStatesTracker(object):
    """
    Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self, tracker_name="model-parallel-rng", mpu=None):
        self.mpu = mpu
        self.tracker_name = tracker_name
        self.states_ = {}
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        return copy.copy(self.states_)

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        self.set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if self.tracker_name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(self.tracker_name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        self.set_cuda_rng_state(self.states_[self.tracker_name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[self.tracker_name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            self.set_cuda_rng_state(orig_cuda_rng_state)

    @staticmethod
    def set_cuda_rng_state(new_state, device=-1):
        """
        Sets the random number generator state of the current GPU.

        Args:
            new_state (torch.ByteTensor): The desired state

        Notes:
            This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
            with a single change: the input state is not cloned. Cloning caused
            major performance issues for +4 GPU cases.
        """
        if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
            # older PyTorch
            def cb():
                with device_ctx_manager(device):
                    _C._cuda_setRNGState(new_state)

        else:
            # newer PyTorch
            if device == -1:
                device = torch.device("cuda")
            elif isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device("cuda", device)

            def cb():
                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                default_generator = torch.cuda.default_generators[idx]
                default_generator.set_state(new_state)

        _lazy_call(cb)
