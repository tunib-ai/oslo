# Features
This document describes various features provided by OSLO briefly.

## 1. Data Parallelism
### 1.1. Distributed Data Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166163261-4c2d5ada-f9bc-44ae-824b-2e2ed82af909.png)
</div>

Distributed Data Parallelism splits the data across multiple GPUs and updates the model on each GPU without parameter servers. It's equal to PyTorch DDP.

#### References
- [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799)
- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)

### 1.2. Sequence Data Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166163539-c51f4bf2-0713-4162-add4-b0b1405a9164.png)
</div>

**Sequence Data Parallelism** is a data parallelization technique that divides data into sequence units. 
We also provide autonomous sequence splitter for this.

#### References
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)

### 1.3. ZeRO Data Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166165692-fc230a94-14cd-43e7-9a82-47a48b3dc2ee.png)
</div>

**ZeRO Data Parallelism** partitions the optimizer states, the gradients, and the model parameters along with the dataset. 
This can dramatically reduce GPU memory usage. We also support CPU and SSD offloading.

#### References
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)

## 2. Model Parallelism
### 2.1. Tensor Model Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166164559-618593f0-79fd-440e-b906-31025b5ca6ad.png)
</div>

**Tensor Model Parallelism** makes it possible to train larger models by partitioning the parameter tensors into multiple dimensions.
We also support 2D, 2.5D, and 3D tensor partitioning which make tensor parallel training more efficient unlike Megatron-LM which simply splits parameters into single dimensions such as rows and columns.

#### References
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
- [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
- [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)

### 2.2. Pipeline Model Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166164859-98c339a8-1629-42d4-a733-2d9e4cb9c3a9.png)
</div>

**Pipeline Model Parallelism** partitions model parameters and pipelines GPU computations by splitting mini-batches into micro-batches.
With OSLO, you don't need to implement the model as nn.Sequential because we support whole new partitioning method called inter-module partitioning,
and we also support various scheduling strategies such as Pipedream Flush and GPipe.

#### References
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)
- [Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training](https://arxiv.org/abs/2111.05972)

### 2.3. Expert Model Parallelism

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166165620-f57e996f-7e7a-43e0-8140-a1e86ebbc4e2.png)
</div>

**Expert Model Parallelism** partitions large FFN layers into multiple pieces and makes each piece an expert in a specific domain. 
We should additionally train the gate function layer for this. This mechanism is also called Mixture of Experts.

#### References
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

## 3. Efficient Components 
### 3.1. Kernel Fusion

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166166013-8615e4da-93ba-46b6-b5dd-acc9deb900c7.png)
</div>

Multiple torch kernels create unnecessary memory loads/stores. 
We reduce this inefficiency by **fusing multiple kernels into a single kernel**. 
For this, we provide various custom fused CUDA kernels like `FusedScaleMaskSoftmax`, `FusedAdam`, `FusedLayerNorm` and so on. 
We also provide JIT-compiled kernel fusion.

#### References
- [Kernel Fusion: An Effective Method for Better Power Efficiency on Multithreaded GPU](https://ieeexplore.ieee.org/document/5724850)
- [Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://arxiv.org/abs/2007.00072)

### 3.2. Lazy Initialization
<div align="center">

![](https://user-images.githubusercontent.com/38183241/166166564-3ed630ae-3f0b-4297-b189-3db51b007cad.png)
</div>

Some frameworks often load the model onto the CPU before parallelizing, which is very memory-inefficient in multiprocess programs. 
We provide a technique to dramatically reduce CPU memory consumption by **lazy initialization** of model parameters.

### 3.3. Activation Checkpointing

<div align="center">

![](https://user-images.githubusercontent.com/38183241/166166846-cd0a06a6-3bf9-4b0b-a223-9a3b94818ce5.png)
</div>

Most autograd functions in PyTorch store the return value of Forward Pass for Backward. 
This is what we call Activation. We provide the feature to save GPU memory consumption by **checkpointing**. 
This feature already exists in PyTorch, but we provide additional features to offload activation to CPU or split it across multiple GPUs and eliminate memory fragmentation.

#### References
- [Efficient Rematerialization for Deep Networks](https://proceedings.neurips.cc/paper/2019/file/ffe10334251de1dc98339d99ae4743ba-Paper.pdf)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

### 3.4. Data Processing

<div align="center">

![](https://t1.daumcdn.net/cfile/tistory/23133B4F5754232522)
</div>

When training large-scale models, we use very large data, and this often causes us to run out of CPU memory. 
We solve this problem by **memory mapping** technique that maps only the addresses of the data on disk to virtual memory. 
We also provide data collators for most NLP tasks. We also support Hugging Face Datasets library, 
so if you have dataset in the Dataset Hub, you can use this features very easily.

#### References
- [Apache Arrow](https://arrow.apache.org/)
- [Hugging Face Datastes](https://huggingface.co/docs/datasets/index)

### 3.5. Model Implementations

<div align="center">

![](https://time-to-reinvent.com/wp-content/uploads/2022/02/rectangle_large_type_2_6b3d7a7cdfb3af98774ab76a8aa9ef03.png)
</div>

We provide efficient model implementations with all the efficient techniques used. 
Our model implementations are compatible with the Hugging Face Model Hub. 
In other words, you can download models from the Hub through the `from_pretrained` method, and upload the trained model to the Hub without checkpoint conversion.
We also support trainer class that is similar with Hugging Face Transformers.

#### References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
