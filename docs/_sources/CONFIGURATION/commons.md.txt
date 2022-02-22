# Commons
You can specify your common configuration under `commons` like:

```json
{
  "commons": {
    "force_gpu": "bool",
  }
}
```
### 1. force_gpu: `bool`
- type: bool
- default: True

When calling `oslo.initialize`, OSLO uploads the model to the GPU. 
If this feature is turned off, the user must manually upload the model to the GPU. 
If the model is not on the GPU, unexpected errors could be occur.
