import inspect

from oslo.pytorch.kernel_fusion.params import ModuleManager
from oslo.pytorch.kernel_fusion.dynamic_shapes.replacer import GraphReplacer
from oslo.pytorch.utils.kernel_fusion_mapping import KernelFusionMapping


class FusionManager(ModuleManager):
    def __init__(self, model, fuser, memory_efficient_fusion=False):
        self.model = model
        self.fuser = fuser
        self.memory_efficient_fusion = memory_efficient_fusion

        mapping = KernelFusionMapping()
        self.params_cls = mapping.get_mapping(model)
        self.supported_args = self.params_cls(model).supported_args()

    def register(self):
        self.model.forward = self._create_new_model_forward(self.model.forward)

    def _create_new_model_forward(self, orig_forward):
        def new_model_forward(*args, **kwargs):
            orig_forward_parameters = inspect.signature(orig_forward).parameters
            param_dict = self.get_param_dict(
                *args, **kwargs, orig_forward_parameters=orig_forward_parameters
            )
            non_default_params = {
                k: v[0] for k, v in param_dict.items() if v[1] is False
            }

            for non_default_name, non_default_value in non_default_params.items():
                if non_default_name not in self.supported_args:
                    if not hasattr(self.model.config, non_default_name):
                        raise ValueError(
                            f"An argument``{non_default_name}`` is not supported in Fused{self.model.__class__.__qualname__}. "
                            f"Currently, only supports {self.supported_args}."
                        )

            input_tensor = kwargs.get("input_ids", None)
            if input_tensor is None:
                input_tensor = kwargs.get("hidden_states", None)
            if input_tensor is None:
                raise ValueError(
                    "forward arguments must be contains ``input_ids`` or ``hidden_states``."
                )

            if not hasattr(self.model, "is_fused"):
                batch_size = input_tensor.size(0)
                seq_len = input_tensor.size(1)
                params_obj = self.params_cls(self.model, batch_size, seq_len)
                module_translated_params = params_obj.translated_params()

                graph_replacer = GraphReplacer(self.model, self.fuser)
                graph_replacer.replace_graph(
                    batch_size, seq_len, module_translated_params, self.memory_efficient_fusion
                )
                setattr(self.model, "is_fused", True)

            return orig_forward(*args, **kwargs)

        return new_model_forward
