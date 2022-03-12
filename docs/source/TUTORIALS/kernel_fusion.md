# Kernel Fusion Tutorial
- Kernel fusion is a technology that can improve training or inference speed.
- OSLO kernel fusion supports the following kernel fusion mechanisms.
    - `JIT based fusion`: kernel fusion based on `torch.jit.script`.
    - `Memory efficient fusion`: kernel fusion based on `AOT Autograd` powered by functorch.
    - `Cusom CUDA kernels`: kernel fusion based on handcrafted CUDA kernels.
- The source code of this tutorial can be found [here](https://github.com/tunib-ai/oslo/tree/main/tutorials).

### Table of contents
* [1. JIT based fusion](#jit-based-fusion)
    + [1.1. Initialize input tensor](#initialize-input-tensor)
    + [1.2. Create models for benchmarking](#create-models-for-benchmarking)
    + [1.3. Fuse kernels](#fuse-kernels)
    + [1.4. Warm-up (compiling)](#warm-up-compiling)
    + [1.5. Benchmark](#benchmark)
* [2. Memory efficient fusion](#memory-efficient-fusion)
    + [2.1. Limitation](#limitation)
    + [2.2. Fuse kernels with AOT Autograd](#fuse-kernels-with-aot-autograd)
    + [2.3. Warm-up (compiling)](#id1)
    + [2.4. Benchmark](#id2)
* [3. Custom CUDA kernels](#custom-cuda-kernels)
    + [3.1. Supported kernels](#supported-kernels)
    + [3.2. Initialize input tensor](#id3)
    + [3.3. Create models for benchmarking](#id4)
    + [3.4. Fuse kernels with the custom CUDA kernels](#34-fuse-kernels-with-the-custom-cuda-kernels)
    + [3.5. Warm-up (compiling)](#id5)
    + [3.6. Benchmark](#id6)

## 1. JIT based fusion
How to use the JIT based fusion?

### 1.1. Initialize input tensor 
```python
import torch

BATCH_SIZE, SEQ_LEN = 256, 16
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN).long().cuda()
```

### 1.2. Create models for benchmarking
```python
from transformers import GPT2LMHeadModel

non_oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
oslo_model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
```

### 1.3. Fuse kernels
If `enable` is set to True, JIT based fusion is used by default.

Note that currently JIT based fusion only supports fused activation function like `FusedGELU`.
In fact, more areas could be fused, but we didn't use model specific policies to make it easier to support many models.

```python
import oslo

oslo_model = oslo.initialize(
    oslo_model, config={"kernel_fusion": {"enable": True}},
)
```

### 1.4. Warm-up (compiling)
JIT compiles the graph at runtime. 
So we should do a warm up to not include compile time in our benchmarks.

```python
for _ in range(10):
    non_oslo_model(input_tensor)

for _ in range(10):
    oslo_model(input_tensor)
```

### 1.5. Benchmark
Experimental results show that 25% faster computation is possible using the kernel fusion engine.
However, this may vary depending on the model architecture.
```python
from time import time

start = time()
for _ in range(10):
    non_oslo_model(input_tensor)
print(f"non-oslo: {time() - start}")


start = time()
for _ in range(10):
    oslo_model(input_tensor)
print(f"oslo: {time() - start}")
```
```
non-oslo: 0.25797200202941895
oslo: 0.20798110961914062
```

## 2. Memory efficient fusion
How to use the memory efficient fusion?

The memory efficient fusion is a kernel fusion mechanism that uses the AOT Autograd engine, a novel engine developed by the functorch team at PyTorch.
The AOT Autograd fuses all fusible areas of the model and also optimizes the backward graph with a novel mechanism called [min-cut rematerialization](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467). 
Because the backward graph can be optimized, the memory efficient fusion shows a huge performance boost in training rather than inference.

However, the AOT Autograd is still under development, so unexpected bugs could be occurred.
In that case, please report the bug to the issue tracker of [OSLO](https://github.com/tunib-ai/oslo/issues) or [functorch](https://github.com/pytorch/functorch/issues).

### 2.1. Limitation
The AOT Autograd currently has the following limitations:
1. Incompatible with model parallelism 
2. Incompatible with activation checkpointing
3. Incompatible with `GenerationMixin` (no `generate` method)
4. Requires PyTorch 1.9+

### 2.2. Fuse kernels with AOT Autograd
Compared to the JIT based fusion section, all other parts are the same. Only this part is different.

```python
import oslo

oslo_model = oslo.initialize(
    oslo_model, 
    config={
        "kernel_fusion": {
            "enable": True, 
            "memory_efficient_fusion": True,
        },
    }
)
```

### 2.3. Warm-up (compiling)
```python
# Warm-up
for _ in range(10):
    non_oslo_model(input_tensor)

for _ in range(10):
    oslo_model(input_tensor)
```

### 2.4. Benchmark
```python
from time import time

# Bench mark
start = time()
for _ in range(10):
    non_oslo_model(input_tensor)
print(f"non-oslo: {time() - start}")

start = time()
for _ in range(10):
    oslo_model(input_tensor)
print(f"oslo: {time() - start}")
```
```
non-oslo: 0.26519250869750977
oslo: 0.19448089599609375
```

The experimental result shows better performance than simple jit based fusion. 

The memory efficient fusion is the most efficient in training scenarios, so you will be able to train your model much more efficiently then simple jit based fusion.


## 3. Custom CUDA kernels
How to use the custom CUDA kernels based fusion?

OSLO provides several handcrafted custom CUDA kernels. 
Currently, only two kernels are supported, but we will continue to expand these in the future.

### 3.1. Supported kernels
- `FusedRMSNorm`: Efficient RMSNorm kernel, it's available when using the T5.
- `FusedNoRepeatNGram`: Execute ngram blocking in GPU when generating text, it's very effective for large batch text generation.

### 3.2. Initialize input tensor
```python
import torch

BATCH_SIZE, SEQ_LEN = 256, 1
input_tensor = torch.ones(BATCH_SIZE, SEQ_LEN).long().cuda()
```

### 3.3. Create models for benchmarking
In this section, I used the T5 model to use `FusedRMSNorm`.

```python
from transformers import T5ForConditionalGeneration

non_oslo_model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
oslo_model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
```

### 3.4. Fuse kernels with the custom CUDA kernels
Input the names of kernels you want to use into the `custom_cuda_kernels`.
Note that custom cuda kernels are compatible with all other mechanisms like the JIT based fusion and the memory efficient fusion.

```python
import oslo

oslo_model = oslo.initialize(
    oslo_model,
    config={
        "kernel_fusion": {
            "enable": True,
            "custom_cuda_kernels": ["FusedNoRepeatNGram", "FusedRMSNorm"],
        },
    },
)
```

### 3.5. Warm-up (compiling)

```python
for _ in range(10):
    non_oslo_model._partition(input_tensor, no_repeat_ngram_size=3)

for _ in range(10):
    oslo_model._partition(input_tensor, no_repeat_ngram_size=3)
```

### 3.6. Benchmark

```python
from time import time

start = time()
for _ in range(10):
    non_oslo_model._partition(input_tensor, no_repeat_ngram_size=3)
print(f"non-oslo: {time() - start}")

start = time()
for _ in range(10):
    oslo_model._partition(input_tensor, no_repeat_ngram_size=3)
print(f"oslo: {time() - start}")
```
```
non-oslo: 1.1885042190551758
oslo: 0.45142364501953125
```

The example in this section shows a 2x performance gain using two custom CUDA kernels and a JIT based fusion.

This concludes the kernel fusion tutorial. Thank you.
