# Activation Checkpointing
You can specify your own activation checkpointing related configuration under `activation_checkpointing` like:

```json
{
  "activation_checkpointing": {
    "enable": "bool",
    "cpu_checkpointing": "bool",
    "partitioned_checkpointing": "bool",
    "contiguous_checkpointing": "bool"
  }
}
```
### 1. enable: `bool`
- type: bool
- default: False

Enable activation checkpointing.

### 2. cpu_checkpointing: `bool`
- type: bool
- default: False

Enable cpu checkpointing.

### 3. partitioned_checkpointing: `bool`
- type: bool
- default: False

Enable partitioned checkpointing.

Note that this is only available when you are using tensor model parallelism.

### 4. contiguous_checkpointing: `bool`

Enable contiguous checkpointing.

Note that this is only available when you are using partitioned checkpointing.

