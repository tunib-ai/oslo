import inspect
from copy import deepcopy

from oslo.pytorch.kernel_fusion.graphs.replacer import GraphReplacer
from oslo.pytorch.kernel_fusion.params import ModuleManager
from oslo.pytorch.utils.kernel_fusion_mapping import KernelFusionMapping


class GraphRegister(ModuleManager):
    def __init__(self, model, fuser, memory_efficient_fusion=False):
        self.model = model
        self.fuser = fuser
        self.memory_efficient_fusion = memory_efficient_fusion
        self.is_fused = False

        mapping = KernelFusionMapping()
        self.params_cls = mapping.get_mapping(model)
        self.supported_args = self.params_cls(model).supported_args()
        self.module2signature = None

    def register(self):
        if not self.is_fused:
            self.model.forward = self._create_new_model_forward(self.model.forward)
            self.is_fused = True

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

            batch_size = input_tensor.size(0)
            seq_len = input_tensor.size(1)
            params_obj = self.params_cls(self.model, batch_size, seq_len)
            module_translated_params = params_obj.translated_params()

            if self.module2signature is None:
                self.module2signature = {}
                for module, translated_params in module_translated_params.items():
                    self.module2signature[translated_params.key()] = deepcopy(
                        dict(inspect.signature(module.forward).parameters)
                    )

            graph_replacer = GraphReplacer(
                self.model,
                self.fuser,
            )

            graph_replacer.replace_graph(
                batch_size,
                seq_len,
                module_translated_params,
                self.module2signature,
                self.memory_efficient_fusion,
            )

            return orig_forward(*args, **kwargs)

        return new_model_forward
