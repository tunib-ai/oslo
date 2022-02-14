# Kernel Fusion
You can specify your own kernel fusion related configuration under `kernel_fusion` like:

```json
{
  "kernel_fusion": {
    "enable": "bool",
    "memory_efficient_fusion": "bool",
    "custom_cuda_kernels": "list",
  }
}
```
### 1. enable: `bool`
- type: bool
- default: False

Enable kernel fusion.

### 2. memory_efficient_fusion: `bool`
- type: bool
- default: False

Enable memory efficient fusion.

### 3. custom_cuda_kernels: `list`
- type: list
- default: []

List of the custom CUDA kernels to use.

Currently, the following kernels are supported.

- `FusedRMSNorm`: Efficient RMSNorm kernel, it's available when using the T5.
- `FusedNoRepeatNGram`: Execute ngram blocking in GPU when generating text, it's very effective for large batch text generation.
 