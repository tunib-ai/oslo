# Tensor Model + Data Parallelism Tutorial
- If you haven't read the [Tensor model parallelism tutorial](https://tunib-ai.github.io/oslo/TUTORIALS/tensor_model_parallelism.html), please read that first.
- The concepts of the tensor model parallelism and data parallelism can be found [here](https://huggingface.co/docs/transformers/parallelism).
- The source code of this tutorial can be found [here](https://github.com/tunib-ai/oslo/tree/main/tutorials).

### Table of contents
* [0. Distributed Launcher](#0-distributed-launcher)
* [1. Multi-dimensional Parallel Training](#1-multi-dimensional-parallel-training)
    + [1.1. The concept of multi-dimensional parallel training](#11-the-concept-of-multi-dimensional-parallel-training)
    + [1.2. The concept of MPU (Model Parallel Unit)](#12-the-concept-of-mpu--model-parallel-unit-)
* [2. Training](#2-training)
    + [2.1. Initialize some variables](#21-initialize-some-variables)
    + [2.2. Create model and optimizer and tokenizer](#22-create-model-and-optimizer-and-tokenizer)
    + [2.3. Parallelize the model](#23-parallelize-the-model)
    + [2.4. Make the model data parallelizable](#24-make-the-model-data-parallelizable)
    + [2.5. Load dataset](#25-load-dataset)
    + [2.6. Create DistributedSampler to parallelize dataset](#26-create-distributedsampler-to-parallelize-dataset)
    + [2.7. Create the dataloader with sampler.](#27-create-the-dataloader-with-sampler)
    + [2.8. Do training as usual](#28-do-training-as-usual)
* [3. Merging Checkpoints](#3-merging-checkpoints)
    + [3.1. Create model](#31-create-model)
    + [3.2. Parallelize the model](#32-parallelize-the-model)
    + [3.3 Load parallelized checkpoints](#33-load-parallelized-checkpoints)
    + [3.4. Merge parallelized checkpoints](#34-merge-parallelized-checkpoints)

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

## 1. Multi-dimensional Parallel Training
### 1.1. The concept of multi-dimensional parallel training

Let's suppose we have 4 GPUs. 

The model parallelism splits the model into multiple pieces and trains a single batch of data.
If we use all these GPUs for model parallelism, It would look like the following.

![image](https://user-images.githubusercontent.com/38183241/151702797-6432b656-7da1-459b-adc5-c0534c3470d3.png)

The data parallelism copies the model into each device and splits the data into multiple pieces and trains the multiple batches of data.
If we use all these GPUs for data parallelism, It would look like the following.

![image](https://user-images.githubusercontent.com/38183241/151702940-0dbbfaa5-2f03-4891-a326-6b28a0359475.png)

Now, we'll mix these two parallelization methods.
We first split the model into multiple pieces. 

![image](https://user-images.githubusercontent.com/38183241/151703444-09c1cdfc-805c-416a-9fa6-06cac5643664.png)

And we now replicate the parallelized model to different GPUs to apply data parallelism.
They have coordinates such as (0, 0), (0, 1), (1, 0), (1, 1) with respect to (data, model) rank.
For this reason, we call this training mechanism 'multi-dimensional parallel training'.

![image](https://user-images.githubusercontent.com/38183241/151708904-75653d57-2a04-4f7c-a712-e182cc56725f.png)

We can make some 'groups' to communicate easily for model and data parallelism.

Model parallel communication is that sends and receives the results of segmented models,
and data parallel communication is that sends and receives results of segmented data.
Therefore, these communications must only take place inside the group.

![image](https://user-images.githubusercontent.com/38183241/151709418-7a59a498-15e9-4e0f-b191-8cf4d2ac9d67.png)

### 1.2. The concept of MPU (Model Parallel Unit)

So it would be nice to use the concept of MPU (Model Parallel Unit) to easily manage these communications. MPU was introduced by NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and OSLO borrowed this concept.
OSLO's MPU provides the following methods to facilitate 3D parallel communication, including pipeline model parallelism, a concept to be introduced later.

```python
from oslo.pytorch.model_parallelism.network.mpu import MPU

mpu = MPU(tensor_parallel_size=4, pipeline_parallel_size=2)

# Data parallel group, rank and world size
mpu.get_data_parallel_group()
mpu.get_data_parallel_rank()
mpu.get_data_parallel_world_size()

# Tensor model parallel group, rank and world size
mpu.get_tensor_parallel_group()
mpu.get_tensor_parallel_rank()
mpu.get_tensor_parallel_world_size()

# Pipeline model parallel group, rank and world size
mpu.get_pipeline_parallel_group()
mpu.get_pipeline_parallel_rank()
mpu.get_pipeline_parallel_world_size()
```

When you use the `oslo.initialize(...)` function, the MPU is created automatically, and the model has its own mpu object.

```python
import oslo

model = oslo.initialize(...)

# Data parallel group, rank and world size
model.mpu.get_data_parallel_group()
model.mpu.get_data_parallel_rank()
model.mpu.get_data_parallel_world_size()
...
```

Let's finish the explanation of multi-dimensional parallelism, and actually use this mechanism to train a model.

## 2. Training
How to use the tensor model + data parallelism for training?

### 2.1. Initialize some variables
```python
BATCH_SIZE = 4
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 100
```

### 2.2. Create model and optimizer and tokenizer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam

model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add pad token for batch training (GPT2 tokenizer doesn't have pad token)
tokenizer.pad_token = tokenizer.eos_token
```

### 2.3. Parallelize the model
Note that `tp size`  * `dp size` must be same or smaller than total num of gpus.
If you have 4 GPUs, you can set `tp size = 2` * `dp size = 2`.
If you specify the `tp size`, the `dp size` will be determined automatically.
```python
import oslo

model = oslo.initialize(
    model, config={"model_parallelism": {"enable": True, "tensor_parallel_size": YOUR_TENSOR_PARALLEL_SIZE}}
)
```

### 2.4. Make the model data parallelizable

You can use torch DDP module with OSLO model parallelism, and you can access the process groups, world sizes and ranks using ``model.mpu``.
For more information about MPU, refer to [here](https://github.com/tunib-ai/oslo/blob/master/oslo/pytorch/model_parallelism/network/mpu.py).
If you are unfamiliar with `DistributedDataParallel`, please refer to [here](https://pytorch.org/tutorials/beginner/dist_overview.html#torch-nn-parallel-distributeddataparallel).

```python
import torch
from torch.nn.parallel import DistributedDataParallel

engine = DistributedDataParallel(
    module=model,
    process_group=model.mpu.get_data_parallel_group(),
    device_ids=[torch.cuda.current_device()],
    output_device=torch.cuda.current_device(),
)
```

### 2.5. Load dataset
I used the Hugging Face `datasets` library in this tutorial.

```python
from datasets import load_dataset

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
```

### 2.6. Create DistributedSampler to parallelize dataset
You must specify the ``num_replicas`` and ``rank`` using ``model.mpu`` when you are creating sampler.
If you are unfamiliar with `DistributedSampler`, please refer to [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler).
```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset=datasets,
    shuffle=True,
    num_replicas=model.mpu.get_data_parallel_world_size(),
    rank=model.mpu.get_data_parallel_rank(),
)
```

### 2.7. Create the dataloader with sampler.
Note that you should turn off shuffle of data loader.
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    datasets,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=sampler,
)
```

### 2.8. Do training as usual
Now that we're all ready, it's time to begin training.
The training code is the same as the previous [Tensor Model Parallelism tutorial](https://tunib-ai.github.io/oslo/TUTORIALS/model_parallelism.html#save-the-parallelized-model). 
However, note that when input batch is forwarding, the DDP engine object is used not the model object,

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

    # Forward-Backward-Step with DDP engine
    # YOU MUST USE ``DDP ENGINE``, NOT ``MODEL`` WHEN YOU ARE FORWARDING INPUT.
    loss = engine(**input_batch, labels=input_batch["input_ids"]).loss
    loss.backward()
    optimizer.step()

    # Save parallelized model
    # This is same with 
    if step % SAVE_INTERVAL == 0:
        model.save_parallelized(save_directory="./parallel_ckpt")

    if step > TRAIN_STEP:
        break
```


## 3. Merging Checkpoints
How to merge the parallelized checkpoints?

The Merging Checkpoints section is same with the Tensor Model Parallelism tutorial. So, if you have already seen the tutorial, you can skip this.

### 3.1. Create model
Usually we create a GPT2 model like this:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
```
However, it is okay to create the randomly initialized model because we will load the local checkpoints after creation.
Here's how to crate a randomly initialized model:
```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)
```

### 3.2. Parallelize the model
```python
import oslo

model = oslo.initialize(
    model, config={"model_parallelism": {"enable": True, "tensor_parallel_size": NUM_YOUR_GPUS}}
)
```

### 3.3 Load parallelized checkpoints
We support `from_parallelized` method to load parallelized checkpoints.
You can load them by just input the save path of parallelized checkpoints.
```python
model = model.from_parallelized("./parallel_ckpt")
```

### 3.4. Merge parallelized checkpoints
The `save_parallelized` method have a special argument named ``merge_checkpoints``.
If you set this argument as Ture, the parallelized checkpoints of model will be saved as merged form.
We recommend merging them after training because this process is a bit slow.
```python
model.save_parallelized("./merged_ckpt", merge_checkpoints=True)
```
```
// merged_ckpt

pytorch_model.bin    config.json
```

This concludes the tensor model + data parallelism tutorial. Thank you.
