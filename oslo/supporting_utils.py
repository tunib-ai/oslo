from contextlib import suppress


LAYER_INFOS = {}

SUPPORTED_MODELS = [
    "Bert",
    "GPT2",
    "T5",
]


def add_model_to_supported_models(model):
    package_name = model.lower()
    model_class_name = model + "PreTrainedModel"
    info_class_name = model + "LayerInfo"

    with suppress(Exception):
        exec(
            f"""from transformers import {model_class_name}\nfrom oslo.models.{package_name} import {info_class_name}\nLAYER_INFOS[{model_class_name}] = [{info_class_name}]"""
        )


if len(LAYER_INFOS) == 0:
    for model in SUPPORTED_MODELS:
        add_model_to_supported_models(model)
