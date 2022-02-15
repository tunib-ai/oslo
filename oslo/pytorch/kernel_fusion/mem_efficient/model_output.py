from dataclasses import asdict, is_dataclass

import torch
from torch.utils import _pytree as pytree


class OutputManager(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def build_model_output_class(cls, fields, values):
        model_output_obj = cls()
        for key, val in zip(fields, values):
            if isinstance(val, torch.Tensor):
                setattr(model_output_obj, key, val)
        return model_output_obj

    def register_model_output_classes(self):
        from transformers.file_utils import ModelOutput
        from transformers.modeling_utils import get_parameter_device

        output = self.model(
            **{
                k: v.to(get_parameter_device(self.model))
                for k, v in self.model.dummy_inputs.items()
            },
            return_dict=True,
        )

        if is_dataclass(output) and isinstance(output, ModelOutput):
            fields = asdict(output.__class__()).keys()
            pytree._register_pytree_node(
                output.__class__,
                lambda x: (
                    [getattr(x, f) for f in fields if getattr(x, f) is not None],
                    None,
                ),
                lambda values, _: self.build_model_output_class(
                    output.__class__, fields, values
                ),
            )
