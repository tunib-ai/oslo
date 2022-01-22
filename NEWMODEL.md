# How to add a new model to OSLO?
This document contains a tutorial about new model adding with BERT model on Hugging Face Transformers.
I didn't originally consider users adding models because this process is quite complex and knowledge of model parallelism is required.
And I don't think users need to know complicated things.
Therefore, I didn't write these types of document. 
However, there were [some recent requests to add models](https://github.com/tunib-ai/oslo/issues/10), 
and I decided to write this document for them. 
If this document is not sufficient, or you encounter errors, 
please comment on the this [issue](https://github.com/tunib-ai/oslo/issues/10).

# Prerequisite
## 1. Copy all the modeling and configuration code from the Hugging Face Transformers
Make a model directory in the `oslo/models` and copy all the modeling and configuration code from the Hugging Face Transformers.
I made `oslo/models/bert/` and copied the `modeling_bert.py` and `configuration_bert.py` from [here](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert).

- modeling_bert.py
```python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

# ... The rest is omitted...
```

- configuration_bert.py
```python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT model configuration"""
from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

# ... The rest is omitted...
```

## 2. Change all the `from ...xxx` to `from transformers.xxx`.
We can't use relative importing because OSLO doesn't exist inside the Transformers. 
So we must change `from ...xxx` to `from transformers.xxx`.

# Tensor Parallelism
The following are the tasks required for the model to be tensor-parallelized.
The explanation of tensor parallelism can be found [here](https://huggingface.co/docs/transformers/master/en/parallelism).

## 1. Write layer policy object.
Layer policy is used to obtain information of the layers. You have to define policy class in the `configuration_xxx.py`.
In general, one model has one policy, but in the case of sequence to sequence models, the encoder and decoder could use different policies.
More examples can be found at [Parallelformers](https://github.com/tunib-ai/parallelformers/blob/main/parallelformers/policies/bert.py) and [DeepSpeed Inference](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py). 
In addition, [this document](https://github.com/tunib-ai/parallelformers/blob/main/POLICY.md) also explains about the policy object.
Please check out the other model's policy object, it could be a good example.
You can find the abstract class in the [`oslo/parallelism/policy.py`](https://github.com/tunib-ai/oslo/blob/master/oslo/parallelism/policy.py).

### 1.1. What is the `Layer` object?
A `Layer` object contains information about an individual layer and this is used in the policy class.
The `Layer` class is defined in [`oslo/parallelism/policy.py`](https://github.com/tunib-ai/oslo/blob/master/oslo/parallelism/policy.py).
You just need to fill in some information according to the form of this object.

```python
@dataclass
class Layer:
    """Data class to describe a layer in the model"""

    module: nn.Module = None
    weight: torch.Tensor = None
    bias: torch.Tensor = None
    replace: dict = None
    n_fused: int = None
    reversed: bool = None
    parallel: bool = True
    tied_embedding: nn.Module = None
```

#### 1.1.1. `module`
Just input a layer module. Here, `layer` means [`BertLayer`](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L445) object.
For example, the attention query projection module of the BERT model is `layer.attention.self.query`.

#### 1.1.2. `weight`
Input a weight parameter of the layer module. 
In the case of the attention query projection weight parameter of the BERT model is `layer.attention.self.query.weight`.
If this layer does't have a weight parameter, you don't need to worry about this. 
Since the default value is set to `None`, it will be automatically set to `None`.

#### 1.1.3. `bias`
Input a bias parameter of the layer module.
In the case of the attention query projection bias parameter of the BERT model is `layer.attention.self.query.bias`.
If this layer does't have a weight parameter, you don't need to worry about this. 
Since the default value is set to `None`, it will be automatically set to `None`.

#### 1.1.4. `replace`
Input a dictionary that contains information about module replacement.
The key of the dictionary have to be a module before replacement, 
and the value of the dictionary have to be a module after replacement.

##### Which module should be replaced?
We should follow the mechanism that proposed from Megatron-LM.

- Attention key, query, value projection layer: `nn.Linear` → `ColumnParallelLinear`
- Attention output projection layer: `nn.Linear` → `RowParallelLinear`
- MLP input projection layer: `nn.Linear` → `ColumnParallelLinear`
- MLP output projection layer: `nn.Linear` → `RowParallelLinear`
- Word embedding layer: `nn.Embedding` → `VocabParallelEmbedding`

For example, in the case of the attention query projection layer module, we can input a dictionary like `{nn.Linear: ColumnParallelLinear}`.

#### 1.1.5 `n_fused`
There are some models that has the combined (fused) attention query, key, value projection parameters in the Transformers.
For example, there is `c_attn` layer module in the `GPT2Model` in the Transformers.
This layer module has the size like (3 * dim, dim). We defined this type of module as the 'fused' module.
So, if the layer module is the 'fused' module, you have to set this argument to the number of fused layer.
In th most cases this value is 1 (not-fused). If the query, key, value parameters are fused, this value should be 3.
If the key and value parameters are fused, this value should be 2.

#### 1.1.6 `reversed`
Normally `nn.Linear` layer module has a weight parameter that has the size of (output, input).
For example, if you created `nn.Linear(256, 512)`, the module has the weight parameter that has the size of (512, 256).

But there is `Conv1D` layer module in the Transformers, and this module has the 'reversed' parameters, 
and this type of parameters have the size of (input, output).
For example, if you create `Conv1D(256, 512)`, the module has the parameter that has the size of (256, 512).
We defined this type of module (like Conv1D) as 'reversed' module.

The fact is that `nn.Linear` layer has the reversed parameters,
but here, we defined the module has the parameters as opposed to `nn.Linear` as the 'reversed' module.
So, if the layer module is the 'reversed' module, you have to set this argument to `True`.

#### 1.1.7 `parallel`
In the Megatron-LM, there are parallelizable parameters and non-parallelizable parameters.
Normally, we parallelize the attention layer, mlp layer and word embedding layer. 
And we do not parallelize the normalization layer and positional embedding layer.
So, if the layer is non-parallelizable, you have to set this argument to `False`.
If you don't have the knowledge about this, I recommend reading their paper, [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053).

#### 1.1.8 `tied_embedding`
In some cases, the head layer could be tied with an input embedding. 
In this case, you need to specify the tied input embedding together when defining the head layer.
For example, the parameter of head layer of the `GPT2LMHead` is `model.lm_head.weight`, and this is tied with `model.transformer.wte.weight`.
In this case, you need to specify the tied input embedding like `Layer(..., tied_embedding=model.transformer.wte.weight)`.
In addition, if the input embedding is parallelized, the tied head layer is already parallelized. So in that case set `parallel` to `False` like `Layer(..., parallel=False)`.

### 1.2. Description of the functions of the Policy object.
In this chapter, I will describe each function of the Policy object.

#### 3.1.1. `reduce_arguments(layer, world_size, config)`
This function adjusts some variables that exist in the model. 
When we perform Tensor model parallelization, the head size and hidden size in the attention layer must be reduced by the world size. 
This function is the function that does this.

It's best to run your code and test exactly which elements you need to reduce. 
However, I have already tested almost all models while making Parallelformers, 
so please refer to the code in Parallelformers.

```python
from oslo.parallelism.mpu import Layer, LayerPolicy


class BertLayerPolicy(LayerPolicy):
    @staticmethod
    def reduce_arguments(layer, world_size, config):
        layer.attention.self.all_head_size = config.hidden_size // world_size,
        layer.attention.self.num_attention_heads = config.num_attention_heads
```

Some models, such as BERT, have conditional initialization layers such as `crossattention`. 
For these layers, you need to use contextlib's `suppress` function to prevent errors.

```python
from oslo.parallelism.mpu import Layer, LayerPolicy
from contextlib import suppress

class BertLayerPolicy(LayerPolicy):
    @staticmethod
    def reduce_arguments(layer, world_size, config):
        layer.attention.self.all_head_size = config.hidden_size // world_size
        layer.attention.self.num_attention_heads = config.num_attention_heads
        
        with suppress(Exception):
            layer.crossattention.self.all_head_size = config.hidden_size // world_size
            layer.crossattention.self.num_attention_heads = config.num_attention_heads // world_size
```

### 3.1.2. `fused_modules()`
This function is used for kernel fusion. I'll explain more detail about this function later. Let's skip this part now.

### 3.1.3. `attn_qkv(layer, config)`
In this function, we register attention query, key, and value parameters.
These functions should return a list of `Layer` objects. I will briefly explain the `Layer` object.
