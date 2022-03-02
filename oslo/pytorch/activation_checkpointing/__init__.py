import importlib

import torch


def initialize_activation_checkpointing(model, config, **kwargs):
    if "activation_checkpointing" in config:
        from oslo.pytorch.activation_checkpointing.checkpoint_engine import (
            ActivationCheckpointingEngine,
        )

        ac_config = config["activation_checkpointing"]
        force_gpu = config["commons"]["force_gpu"]

        if "enable" in ac_config and ac_config["enable"] is True:
            mpu = model.mpu if hasattr(model, "mpu") else None

            if mpu is None or mpu.get_tensor_parallel_world_size() <= 1:
                partitioned_checkpointing = ac_config.get(
                    "partitioned_checkpointing", False
                )
                contiguous_checkpointing = ac_config.get(
                    "contiguous_checkpointing", False
                )
                assert not partitioned_checkpointing and not contiguous_checkpointing, (
                    "``partitioned_checkpointing`` and ``contiguous_checkpointing`` can be used when you are used tensor model parallelism. "
                    "please set them as False if you don't use tensor model parallelism."
                )

            if ac_config.get("contiguous_checkpointing", False):
                assert ac_config.get("partitioned_checkpointing", False), (
                    "``contiguous_checkpointing`` is only available with ``partitioned_checkpointing``. "
                    "Set ``partitioned_checkpointing`` to true in your config."
                )

            if force_gpu is True and model.device == torch.device("cpu"):
                model = model.cuda()

            engine = ActivationCheckpointingEngine(
                mpu=mpu,
                num_layers=model.config.num_hidden_layers,
                partitioned_checkpointing=ac_config.get(
                    "partitioned_checkpointing", False
                ),
                cpu_checkpointing=ac_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=ac_config.get(
                    "contiguous_checkpointing", False
                ),
            )

            module = importlib.import_module(model.__module__)

            for name, prop in module.__dict__.items():
                if prop is torch:
                    getattr(
                        module, name
                    ).utils.checkpoint.checkpoint = engine.checkpoint
                elif prop is torch.utils:
                    getattr(module, name).checkpoint.checkpoint = engine.checkpoint
                elif prop is torch.utils.checkpoint:
                    getattr(module, name).checkpoint = engine.checkpoint
                elif prop is torch.utils.checkpoint.checkpoint:
                    setattr(module, name, engine.checkpoint)

            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            model = model.train()

    return model, config
