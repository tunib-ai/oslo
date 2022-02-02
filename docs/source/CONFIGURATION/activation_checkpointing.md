# Model Parallelism
You can specify your own activation checkpointing related configuration under `model_parallelism` like:

```json
{
  "activation_checkpointing": {
    "enable": bool,
    "backend": "torch" | "fairscale" | "deepspeed",
    "fairscale_params": {
      "offload_to_cpu": bool
    },
    "deepspeed_params": {
      
    }
  }
}
```
### 1. enable: `bool`
- type: bool
- default: False

Enable activation checkpointing

### 2. backend: `str`
- type: str
- default: "torch"
- available: ["torch", "fairscale", "deepspeed"]

Select backend engine for activation checkpointing

### 3. fairscale_params: `dict`
Additional parameters for fairscale activation checkpointing

#### `offload_to_cpu`: `bool`


---


### `tensor_parallel_size`: `int`
- type: int
- default: 1

This means tensor model parallelism degree. If you don't know what tensor model parallelism is,
you can find the detail of it from [here](https://huggingface.co/docs/transformers/parallelism).

Note there are some rules for the tensor parallel size.

- ``tensor_parallel_size`` must be same or smaller than total number of gpus.
- ``tensor_parallel_size`` must be power of 2. (e.g. 2, 4, 8, 16, ...)
- ``tensor_parallel_size`` must be positive number.

#### SUPPORTED MODELS
The following list includes currently supported models.

- Albert
- Bert
- Bart
- T5
- GPT2
- GPTNeo
- GPTJ
- Electra
- Roberta


We will support almost all models in the near future ;)
