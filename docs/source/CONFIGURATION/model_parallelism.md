# Model Parallelism
You can specify your own model parallelism related configuration under `model_parallelism` like:

```json
{
  "model_parallelism": {
    "enable": bool,
    "tensor_parallel_size": int,
  }
}
```
### 1. enable: `bool`
- type: bool
- default: False

Enable model parallelism

### 2. tensor_parallel_size: `int`
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
