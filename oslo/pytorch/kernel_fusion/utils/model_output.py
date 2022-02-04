from contextlib import suppress
from dataclasses import asdict, is_dataclass
from torch.utils import _pytree as pytree


def build_model_output_class(cls, fields, values):
    model_output_obj = cls()
    for key, val in zip(fields, values):
        setattr(model_output_obj, key, val)

    return model_output_obj


def register_model_output_classes(model):
    from transformers.file_utils import ModelOutput
    from transformers.modeling_utils import get_parameter_device

    output = model(
        **{k: v.to(get_parameter_device(model)) for k, v in model.dummy_inputs.items()},
        return_dict=True,
    )

    if is_dataclass(output) and isinstance(output, ModelOutput):
        fields = asdict(output.__class__()).keys()
        pytree._register_pytree_node(
            output.__class__,
            lambda x: ([getattr(x, k) for k in fields], None),
            lambda values, _: build_model_output_class(
                output.__class__, fields, values
            ),
        )
