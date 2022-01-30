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
*  January 30, 2022 [Released OSLO 2.0 alpha version](https://github.com/tunib-ai/oslo/releases/tag/v2.0.0a0).
* December 30, 2021 [Add Deployment Launcher](https://github.com/tunib-ai/oslo/releases/tag/v1.0).
* December 21, 2021 [Released OSLO 1.0](https://github.com/tunib-ai/oslo/releases/tag/v1.0).

## What is OSLO about?
OSLO is a framework that provides various GPU based optimization technologies for large-scale modeling. 
3D Parallelism and Kernel Fusion which could be useful when training a large model like [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) are the key features. OSLO makes these technologies easy-to-use by magical compatibility with [Hugging Face Transformers](https://github.com/huggingface/transformers) that is being considered as a <i>de facto</i> standard in 2021.

## Installation
OSLO can be easily installed using the pip package manager.
All the dependencies such as [torch](https://pypi.org/project/torch/) and [transformers](https://pypi.org/project/transformers/) should be installed automatically with the following command.
Be careful that the 'core' is in the PyPI project name.
```console
pip install oslo-core
```

## Basic Usage
It only takes a single line of code. Now feel free to train and infer a large transformer model. 😎

```python
import oslo

model = oslo.initialize(model, "oslo-config.json")
```

## Documents
For detailed information, refer to [our official document](https://tunib-ai.github.io/oslo/).

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
