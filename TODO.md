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