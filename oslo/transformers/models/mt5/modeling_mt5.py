from oslo.transformers.models.t5.modeling_t5 import (
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Model,
)

try:
    from transformers.models.mt5.configuration_mt5 import MT5Config
    from transformers.utils import logging
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


logger = logging.get_logger(__name__)


class MT5Model(T5Model):
    r"""
     This class overrides [`T5Model`]. Please check the superclass for the appropriate documentation alongside usage
     examples.

    `"""

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides [`T5ForConditionalGeneration`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides [`T5EncoderModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.

    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder.embed_tokens.weight",
    ]
