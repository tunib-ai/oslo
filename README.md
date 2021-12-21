<div align="center">

![](assets/oslo.png)

<br>

## O S L O

**O**pen **S**ource framework for **L**arge-scale transformer **O**ptimization

<p align="center">
<a href="https://github.com/tunib-ai/oslo/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/tunib-ai/oslo.svg" /></a> 
<a href="https://github.com/tunib-ai/oslo/blob/master/LICENSE.apache-2.0"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/></a> <a href="https://tunib-ai.github.io/oslo"><img alt="Docs" src="https://img.shields.io/badge/docs-passing-success.svg"/></a>
<a href="https://github.com/tunib-ai/oslo/issues"><img alt="Issues" src="https://img.shields.io/github/issues/tunib-ai/oslo"/></a>

    
</div>

<br><br>

### What's New:
* December 21, 2021 [Released OSLO 1.0](https://github.com/tunib-ai/oslo/releases/tag/v1.0).

## What is OSLO about?

OSLO is a framework that provides various GPU based optimization features for large-scale modeling. As of 2021, the [Hugging Face Transformers](https://github.com/huggingface/transformers) is being considered <i>de facto</i> standard.
However, it does not best fit the purposes of large-scale modeling yet.
This is where OSLO comes in. OSLO is designed to make it easier to train large models with the Transformers.
For example, you can fine-tune [GPTJ](https://huggingface.co/EleutherAI/gpt-j-6B) on the [Hugging Face Model Hub](https://huggingface.co/models) without many extra efforts using OSLO. Currently, GPT2, GPTNeo, and GPTJ are supported, but we plan to support more soon.

## Installation
OSLO can be easily installed using the pip package manager.
All the dependencies such as [torch](https://pypi.org/project/torch/), [transformers](https://pypi.org/project/transformers/), [dacite](https://pypi.org/project/dacite/),
[ninja](https://pypi.org/project/ninja/) and [pybind11](https://pypi.org/project/pybind11/) should be installed automatically with the following command.
Be careful that the 'core' in the PyPI project name.
```console
pip install oslo-core
```

Some of features rely on the C++ language.
So we provide an option, `CPP_AVAILABLE`, to decide whether or not you install them. 

- If the C++ is available:
```console
CPP_AVAILABLE=1 pip install oslo-core
```

- If the C++ is not available:
```console
CPP_AVAILABLE=0 pip install oslo-core
```

Note that the default value of `CPP_AVAILABLE` is 0 in Windows and 1 in Linux.

## Key Features

```python
import deepspeed 
from oslo import GPTJForCausalLM

# 1. 3D Parallelism
model = GPTJForCausalLM.from_pretrained_with_parallel(
    "EleutherAI/gpt-j-6B", tensor_parallel_size=2, pipeline_parallel_size=2,
)

# 2. Kernel Fusion
model = model.fuse()

# 3. DeepSpeed Support
engines = deepspeed.initialize(
    model=model.gpu_modules(), model_parameters=model.gpu_paramters(), ...,
)

# 4. Data Processing
from oslo import (
    DatasetPreprocessor, 
    DatasetBlender, 
    DatasetForCausalLM, 
    ...    
)
```

OSLO offers the following features.

- **3D Parallelism**: The state-of-the-art technique for training a large-scale model with multiple GPUs.
- **Kernel Fusion**: A GPU optimization method to increase training and inference speed. 
- **DeepSpeed Support**: We support [DeepSpeed](https://github.com/microsoft/DeepSpeed) which provides ZeRO data parallelism.
- **Data Processing**: Various utilities for efficient large-scale data processing.

See [USAGE.md](USAGE.md) to learn how to use them.

## Administrative Notes

### Citing OSLO
If you find our work useful, please consider citing:

```
@misc{oslo,
  author       = {Ko, Hyunwoong and Kim, Soohwan and Park, Kyubyong},
  title        = {OSLO: Open Source framework for Large-scale transformer Optimization},
  howpublished = {\url{https://github.com/tunib-ai/oslo}},
  year         = {2021},
}
```

### Licensing

The Code of the OSLO project is licensed under the terms of the [Apache License 2.0](LICENSE.apache-2.0).

Copyright 2021 TUNiB Inc. http://www.tunib.ai All Rights Reserved.

### Acknowledgements

The OSLO project is built with GPU support from the [AICA (Artificial Intelligence Industry Cluster Agency)](http://www.aica-gj.kr).
