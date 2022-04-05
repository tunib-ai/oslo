from torch.nn.modules.lazy import LazyModuleMixin


class ParentLazyModuleMixin(LazyModuleMixin):
    def initialize_parameters(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, LazyModuleMixin):
                module.initialize_parameters(*args, **kwargs)
            if hasattr(module, "_initialize_hook"):
                module._initialize_hook.remove()
                delattr(module, '_initialize_hook')
            if hasattr(module, "_load_hook"):
                module._load_hook.remove()
                delattr(module, '_load_hook')
