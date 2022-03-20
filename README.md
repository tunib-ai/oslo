<div align="center">

![](https://github.com/tunib-ai/oslo/raw/master/assets/oslo.png)

<br>

## O S L O

**O**pen **S**ource framework for **L**arge-scale transformer **O**ptimization

<p align="center">
<a href="https://github.com/tunib-ai/oslo/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/tunib-ai/oslo.svg" /></a>
<a href="https://github.com/tunib-ai/oslo/blob/master/LICENSE.apache-2.0"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/></a> <a href="https://tunib-ai.github.io/oslo"><img alt="Docs" src="https://img.shields.io/badge/docs-passing-success.svg"/></a>
<a href="https://github.com/tunib-ai/oslo/issues"><img alt="Issues" src="https://img.shields.io/github/issues/tunib-ai/oslo"/></a>


</div>

## Installation
OSLO can be easily installed using the pip package manager.
All the dependencies such as [torch](https://pypi.org/project/torch/) and [transformers](https://pypi.org/project/transformers/) should be installed automatically with the following command.
**Be careful that the 'core' is in the PyPI project name.**

```console
pip install oslo-core
```

## TODO
### 1. TP
- 1D TP ([megatron](https://github.com/NVIDIA/Megatron-LM), but also in the [clossal](https://github.com/hpcaitech/ColossalAI) [ai](https://www.colossalai.org/docs/get_started/installation))
- 2D TP ([clossal](https://github.com/hpcaitech/ColossalAI) [ai](https://www.colossalai.org/docs/get_started/installation))
- 2.5D TP ([clossal](https://github.com/hpcaitech/ColossalAI) [ai](https://www.colossalai.org/docs/get_started/installation))
- 3D TP ([clossal](https://github.com/hpcaitech/ColossalAI) [ai](https://www.colossalai.org/docs/get_started/installation))

### 2. PP
- scheduling
  - pipedraem-flush ([deepspeed](https://github.com/microsoft/DeepSpeed))
  - varuna ([microsoft](https://github.com/microsoft/varuna))
- partitioning
  - inter-layer partitioning ([gpipe](https://github.com/kakaobrain/torchgpipe) or [deepspeed](https://github.com/microsoft/DeepSpeed))
  - inter-module partitioning ([sagemaker](https://arxiv.org/pdf/2111.05972.pdf))

### 3. SP
- Ring self attention ([clossal](https://github.com/hpcaitech/ColossalAI) [ai](https://www.colossalai.org/docs/get_started/installation))

### 4. DP
- DDP ([torch](https://pytorch.org/docs/master/notes/ddp.html))
- ZeRO ([deepspeed](https://github.com/microsoft/DeepSpeed))
  - ZeRO 1, 2, 3 and offloading

### 5. Kernel Fusion
- CUDA based ([apex](https://github.com/NVIDIA/apex), [deepspeed](https://github.com/microsoft/DeepSpeed), [megatron](https://github.com/NVIDIA/Megatron-LM))
  - optimizers (adam, adafactor, ...)
  - layers (norms, softmax, ...)
- Compile based (torch)
  - [torch.jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html) based functions
  - [AOTAutograd](https://pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html) from functorch

### 6. Transformers model implementation
- integration with Hugging Face model hub

### 7. ETC
- [activation checkpointing](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/activation_checkpointing)
- data loader utils
- ...

### Licensing

The Code of the OSLO project is licensed under the terms of the [Apache License 2.0](LICENSE.apache-2.0).

Copyright 2021 TUNiB Inc. http://www.tunib.ai All Rights Reserved.

### Acknowledgements

The OSLO project is built with GPU support from the [AICA (Artificial Intelligence Industry Cluster Agency)](http://www.aica-gj.kr).
