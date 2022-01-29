import json
import os
from typing import Union, Dict, Any

from transformers import PreTrainedModel

from oslo.pytorch.model_parallelism import initialize_model_parallelism

SUPPORTED_FEATURES = {
    "model_parallelism": {
        "tensor_parallel_size": int,
        "from_checkpoints": str,
    },
}


def _config_check(form, val, key=None):
    assert len(val) > 0, f"There are no arguments in dictionary ``{key}``."
    form_type = form if isinstance(form, type) else type(form)
    config_name = val if isinstance(val, str) else key

    assert isinstance(val, form_type), (
        f"An argument ``{config_name}`` must be type of {form_type}. "
        f"but you input {val}."
    )

    if isinstance(val, dict):
        for k in val:
            assert k in form, (
                f"An argument ``{k}`` is not available. "
                f"We only support the arguments like {list(form.keys())}."
            )
            if isinstance(val[k], dict):
                _config_check(form[k], val[k], k)
            else:
                _form_type = form[k] if isinstance(form[k], type) else type(form[k])

                assert isinstance(val[k], _form_type), (
                    f"An argument ``{k}`` must be type of {_form_type}. "
                    f"but you input {type(val[k])}"
                )


def _sanity_check(
    model: PreTrainedModel,
    config: Union[str, Dict[str, Any]],
):
    assert isinstance(
        model, PreTrainedModel
    ), "An argument ``model`` must be the PyTorch Hugging Face Transformers model."

    assert isinstance(config, dict) or isinstance(
        config, str
    ), "An argument ``config`` must be the dictionary object or json path."

    if isinstance(config, str):
        if os.path.isfile(config):
            config = json.load(open(config, encoding="utf-8"))
        else:
            raise ValueError(
                f"Con not find the {config}. "
                f"Please double check your config file name."
            )

    _config_check(SUPPORTED_FEATURES, config)

    return model, config


def initialize(model: PreTrainedModel, config: Union[str, Dict[str, Any]], **kwargs):
    """
    Initialize OSLO engine.

    Args:
        model (PreTrainedModel): The PyTorch Hugging Face Transformers model
        config (Union[str, os.PathLike, Dict[str, Any]]): dict object or json path
    """
    model, config = _sanity_check(model, config)
    model, config = initialize_model_parallelism(model, config, **kwargs)
    return model
