# Tensor Model Parallelism Tutorial
- The concept of the tensor model parallelism can be found [here](https://huggingface.co/docs/transformers/parallelism).
- The source code of this tutorial can be found [here](https://github.com/tunib-ai/oslo/tree/main/tutorials).

### Table of contents
- [0. Distributed Launcher](#distributed-launcher)
- [1. Inference](#inference)
  + [1.1. Create model and tokenizer](#create-model-and-tokenizer)
  + [1.2. Parallelize the model](#parallelize-the-model)
  + [1.3. Do inference as usual](#do-inference-as-usual)
- [2. Training](#training)
  + [2.1. Initialize some variables](#initialize-some-variables)
  + [2.2. Load dataset and create data loader](#load-dataset-and-create-data-loader)
  + [2.3. Create model, optimizer and tokenizer](#create-model--optimizer-and-tokenizer)
  + [2.4. Parallelize the model](#parallelize-the-model)
  + [2.5. Do training as usual](#do-training-as-usual)
  + [2.6. Save the parallelized model](#save-the-parallelized-model)
- [3. Merging Checkpoints](#merging-checkpoints)
  + [3.1. Create model](#create-model)
  + [3.2. Parallelize the model](#parallelize-the-model)
  + [3.3 Load parallelized checkpoints](#load-parallelized-checkpoints)
  + [3.4. Merge parallelized checkpoints](#merge-parallelized-checkpoints)


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

## 1. Inference 
How to use the tensor model parallelism for inference?

### 1.1. Create model and tokenizer
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

### 1.2. Parallelize the model
- ``tensor_parallel_size`` must be same or smaller than total number of gpus.
- ``tensor_parallel_size`` must be power of 2. (e.g. 2, 4, 8, 16, ...)
- ``tensor_parallel_size`` must be positive number.
- ``tensor_parallel_size`` must be same or greater than hidden size
- ``tensor_parallel_size`` must be same or greater than the number of heads

```python
import oslo

model = oslo.initialize(
    model, config={"model_parallelism": {"enable": True, "tensor_parallel_size": NUM_YOUR_GPUS}}
)
```

You can also use json file (example for 4GPUs)

```json
// oslo-config.json

{
     "model_parallelism": {
          "enable": true,
          "tensor_parallel_size": 4
      }
}
```

And you can use the json file like this:

```python
model = oslo.initialize(model, config="oslo-config.json")
```

### 1.3. Do inference as usual
This is an example of text generation.
In addition to this, it can be used in various tasks such as sequence classification or masked lm.
Likewise, you can write the code as usual.
```python
text = "I don't want a lot for Christmas. There is just one thing"
tokens = tokenizer(text, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**tokens, num_beams=3)[0]))
```
```
I don't want a lot for Christmas. There is just one thing I want to ...
```

## 2. Training
How to use the tensor model parallelism for training?

### 2.1. Initialize some variables
```python
BATCH_SIZE = 4
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 100
```

### 2.2. Create model, optimizer and tokenizer
```python
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add pad token for batch training 
# GPT2 tokenizer doesn't have pad token.
tokenizer.pad_token = tokenizer.eos_token
```

### 2.3. Parallelize the model
```python
import oslo

model = oslo.initialize(
    model, config={"model_parallelism": {"tensor_parallel_size": NUM_YOUR_GPUS}}
)
```

### 2.4. Load dataset and create dataloader
In this tutorial, I used `datasets` library of Hugging Face.

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
```

### 2.5. Do training as usual
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
    loss = model(**input_batch, labels=input_batch["input_ids"]).loss
    loss.backward()
    optimizer.step()
```

### 2.6. Save the parallelized model
We support ``save_parallelized`` method, and this is similar with [``save_pretrained``](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained) in the Transformers.
So, it can be used with the same argument with [`save_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained).
Then, the checkpoints like `pytorch_model_tp_0_pp_0.bin` will be saved in your local path.

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
    loss = model(**input_batch, labels=input_batch["input_ids"]).loss
    loss.backward()
    optimizer.step()
    
    # Save the parallelized model using `save_parallelized`
    if step % SAVE_INTERVAL == 0:
        model.save_parallelized(save_directory="./parallel_ckpt")

    if step > TRAIN_STEP:
        break
```

## 3. Merging Checkpoints
How to merge the parallelized checkpoints?

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

This concludes the tensor model parallelism tutorial. Thank you.
