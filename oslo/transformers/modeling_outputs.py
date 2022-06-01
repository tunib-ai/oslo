from dataclasses import dataclass
from collections import OrderedDict


class ModelOutput(OrderedDict):
    """Marker Interface"""


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """Marker Interface"""


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """Marker Interface"""


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """Marker Interface"""


@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    """Marker Interface"""


@dataclass
class TokenClassifierOutput(ModelOutput):
    """Marker Interface"""
