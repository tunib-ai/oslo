import contextlib
import functools

import torch


def _substitute_init_recursively(cls, func):
    for subcls in cls.__subclasses__():
        _substitute_init_recursively(subcls, func)
        func(subcls)


class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self):
        pass

    def __enter__(self):

        def preprocess_after(f):

            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        # Excution self._post_init_method after the default init function.
        _substitute_init_recursively(torch.nn.modules.module.Module, _enable_class)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

    def __exit__(self, exc_type, exc_value, traceback):

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        _substitute_init_recursively(torch.nn.modules.module.Module, _disable_class)

        torch.nn.modules.module.Module.__init_subclass__ = (torch.nn.modules.module.Module._old_init_subclass)

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass


class ZeroInitContext(InsertPostInitMethodToModuleSubClasses):
    def __init__(self):
        super().__init__()

    def _post_init_method(self, module: torch.nn.Module):
        pass
