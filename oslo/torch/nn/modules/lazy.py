from torch.nn.modules.lazy import LazyModuleMixin


class ParentLazyModuleMixin(LazyModuleMixin):
    def initialize_parameters(self, *args, **kwargs):
        for module in self.modules():
            if isinstance(module, LazyModuleMixin):
                module.initialize_parameters(*args, **kwargs)
