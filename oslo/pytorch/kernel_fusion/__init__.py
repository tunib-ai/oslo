import torch

from oslo.pytorch.kernel_fusion.utils.torch_version import higher_than


def initialize_kernel_fusion(model, config, **kwargs):
    if "kernel_fusion" in config:
        kf_config = config["kernel_fusion"]

        if "enable" in kf_config and kf_config["enable"] is True:
            from oslo.pytorch.kernel_fusion.kernel_fusion_engine import (
                KernelFusionEngine,
            )

            if model.device == torch.device("cpu"):
                model = model.cuda()

            memory_efficient_fusion = kf_config.get("memory_efficient_fusion", False)
            custom_cuda_kernels = kf_config.get("custom_cuda_kernels", None)

            if memory_efficient_fusion is True:
                if (
                    "model_parallelism" in config
                    and config["model_parallelism"]["enable"] is True
                ):
                    raise ValueError(
                        "``memory_efficient_fusion`` is not compatible with model parallelism."
                    )
                if (
                    "activation_checkpointing" in config
                    and config["activation_checkpointing"]["enable"] is True
                ):
                    raise ValueError(
                        "``memory_efficient_fusion`` is not compatible with activation checkpointing."
                    )

                if not higher_than(1, 9):
                    raise EnvironmentError(
                        "``memory_efficient_fusion`` is compatible with PyTorch 1.9+"
                    )

            engine = KernelFusionEngine(
                model=model,
                memory_efficient_fusion=memory_efficient_fusion,
                custom_cuda_kernels=custom_cuda_kernels,
            )
            model = engine.fuse()

    return model, config
