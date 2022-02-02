# Activation Checkpointing Tutorial
- If you haven't read the [Tensor model parallelism tutorial](https://tunib-ai.github.io/oslo/TUTORIALS/tensor_model_parallelism.html), please read that first.
- OSLO activation checkpointing is based on PyTorch activation checkpointing and adds CPU checkpointing, Partitioned checkpointing, and Contiguous checkpointing described in the [this paper](https://arxiv.org/abs/1910.02054).
  - `CPU checkpointing`: offloads activation memory to CPU
  - `Partitioned checkpointing`: partitions activation memory into multiple GPUs
  - `Contiguous checkpointing`: avoids activation memory fragmentation
- If you are unfamiliar with activation checkpointing, please see [this](https://pytorch.org/docs/stable/checkpoint.html).
- The source code of this tutorial can be found [here](https://github.com/tunib-ai/oslo/tree/main/tutorials).

### Table of contents
* [0. Distributed Launcher](#0-distributed-launcher)
* [1. Training with PyTorch activation checkpointing](#1-training-with-pytorch-activation-checkpointing)
    + [1.1. Initialize some variables](#11-initialize-some-variables)
    + [1.2. Create model and optimizer and tokenizer](#12-create-model-and-optimizer-and-tokenizer)
    + [1.3. Parallelize the model](#13-parallelize-the-model)
    + [1.4. Enable PyTorch activation checkpointing](#14-enable-pytorch-activation-checkpointing)
    + [1.5. Load dataset and create data loader](#15-load-dataset-and-create-data-loader)
    + [1.6. Do training as usual](#16-do-training-as-usual)
* [2. Training with OSLO activation checkpointing](#2-training-with-oslo-activation-checkpointing)
    + [2.1. Enable OSLO activation checkpointing](#21-enable-oslo-activation-checkpointing)
    + [2.2. Do training as usual](#22-do-training-as-usual)

## 0. Distributed Launcher
This tutorial must be launched using distributed launcher.

If you have 4 GPUs:
```console
python -m torch.distributed.launch --nproc_per_node=4 YOUR_SCRIPT.py
```
If you installed DeepSpeed in your environments, the following works the same.
```colsole
deepspeed --num_gpus=4 YOUR_SCRIPT.py
```
For more information of the distributed launchers, refer to:
- [Pytorch documents](https://pytorch.org/docs/stable/distributed.html)
- [DeepSpeed documents](https://www.deepspeed.ai/getting-started/#launching-deepspeed-training)

## 1. Training with PyTorch activation checkpointing
How to use PyTorch activation checkpointing for training?

### 1.1. Initialize some variables
```python
BATCH_SIZE = 128
SEQ_LEN = 128
TRAIN_STEP = 10
```

### 1.2. Create model and optimizer and tokenizer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam

model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add pad token for batch training (GPT2 tokenizer doesn't have pad token)
tokenizer.pad_token = tokenizer.eos_token
```

### 1.3. Parallelize the model
Note that PyTorch activation checkpointing can be used without model parallelism.
```python
import oslo

model = oslo.initialize(
    model, config={"model_parallelism": {"enable": True, "tensor_parallel_size": YOUR_TENSOR_PARALLEL_SIZE}}
)
```

### 1.4. Enable PyTorch activation checkpointing
The activation checkpointing is implemented in ``torch.utils.checkpoint`` package. 
It is already integrated with the Hugging Face Transformers, so you can enable it using ``model.gradient_checkpointing_enable()``.

```python
model.gradient_checkpointing_enable()
```

### 1.5. Load dataset and create data loader
In this tutorial, I used `datasets` library of Hugging Face.
```python
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
```

### 1.6. Do training as usual
```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # Make batch
    input_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # Forward-Backward-Step
    loss = model(**input_batch, labels=input_batch["input_ids"], use_cache=False).loss
    if torch.distributed.get_rank() == 0:
        print(f"memory: {round(torch.cuda.memory_allocated() / (1024 ** 3), 4)}GiB")
    loss.backward()
    optimizer.step()

    if step > TRAIN_STEP:
        break
```
```
memory: 12.8594 GiB
```

## 2. Training with OSLO activation checkpointing
Most of the code used in `Training with pytorch activation checkpointing` is the same, only the `Enable activation checkpointing` part of 1.4 is changed.

### 2.1. Enable OSLO activation checkpointing
Please initialize oslo engine like the following instead of calling ``model.gradient_checkpointing_enable()``.

```python
model = oslo.initialize(
    model,
    config={
        "model_parallelism": {
            "enable": True,
            "tensor_parallel_size": YOUR_TENSOR_PARALLEL_SIZE,
        },
        "activation_checkpointing": {
            "enable": True,
            "cpu_checkpointing": True,
            "partitioned_checkpointing": True,
            "contiguous_checkpointing": True,
        },
    },
)
```


### 2.2. Do training as usual
```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # Make batch
    input_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # Forward-Backward-Step
    loss = model(**input_batch, labels=input_batch["input_ids"], use_cache=False).loss
    if torch.distributed.get_rank() == 0:
        print(f"memory: {round(torch.cuda.memory_allocated() / (1024 ** 3), 4)}GiB")
    loss.backward()
    optimizer.step()

    if step > TRAIN_STEP:
        break
```
```
memory: 6.681GiB
```
As the result, you can save about twice the memory, so you can train model more efficiently using a larger batch size.

This concludes the activation checkpointing tutorial. Thank you.
