import itertools
from torch.nn.parameter import is_lazy
from torch.nn.modules.lazy import _LazyProtocol


class LazyModuleMixin:
    cls_to_become = None

    def __init__(self: _LazyProtocol, *args, **kwargs):
        # Mypy doesn't like this super call in a mixin
        super().__init__(*args, **kwargs)  # type: ignore[misc]

    def has_uninitialized_params(self: _LazyProtocol):
        r"""Check if a module has parameters that are not initialized"""
        # This is to avoid the JIT to track this parameter and force
        # custom modules __setstate__ to add it
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_lazy(param):
                return True
        return False

    def initialize_parameters(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, LazyModuleMixin):
                module.initialize_parameters(*args, **kwargs)
