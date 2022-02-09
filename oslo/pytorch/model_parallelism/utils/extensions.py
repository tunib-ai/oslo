import os
from collections import Callable
from logging import getLogger
from typing import Optional, Union

import torch
import torch.distributed as dist

from oslo.pytorch.model_parallelism.tensor_parallel_enigne import (
    TensorDeparallelEngine,
    TensorParallelEngine,
)

PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"

logger = getLogger(__name__)


def from_parallelized(parallelized_model_path, **kwargs):
    self = kwargs.pop("self")

    file_names = {
        os.path.join(
            parallelized_model_path,
            PARALLELIZED_WEIGHTS_NAME.replace("tp_0", f"tp_{tp}").replace(
                "pp_0", f"pp_{pp}"
            ),
        )
        for tp in range(self.mpu.get_tensor_parallel_world_size())
        for pp in range(self.mpu.get_pipeline_parallel_world_size())
    }

    if os.path.isdir(parallelized_model_path):
        if all(os.path.isfile(file_name) for file_name in file_names):
            state_dict = torch.load(
                os.path.join(
                    parallelized_model_path,
                    PARALLELIZED_WEIGHTS_NAME.replace(
                        "tp_0",
                        f"tp_{self.mpu.get_tensor_parallel_rank()}",
                    ).replace(
                        "pp_0",
                        f"pp_{self.mpu.get_pipeline_parallel_rank()}",
                    ),
                )
            )

            if self._keys_to_ignore_on_save is not None:
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k not in self._keys_to_ignore_on_save
                }

            self.load_state_dict(state_dict, strict=False)

        else:
            raise FileNotFoundError(
                f"all the {file_names} are necessary. "
                f"but some of them do not exist. Please check your checkpoint files."
            )
    else:
        raise NotADirectoryError(
            f"directory named {parallelized_model_path} is not valid. "
        )

    return self


@torch.no_grad()
def save_parallelized(
    save_directory: Union[str, os.PathLike],
    save_config: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    merge_checkpoints: bool = False,
    **kwargs,
):
    from transformers.modeling_utils import get_parameter_dtype, unwrap_model

    self = kwargs.pop("self")
    mapping = kwargs.pop("tp_mapping", None)

    if (
        self.mpu.get_tensor_parallel_world_size() == 1
        and self.mpu.get_pipeline_parallel_world_size() == 1
    ):
        if dist.get_rank() == 0:
            self.save_pretrained(
                save_directory=save_directory,
                save_config=save_config,
                state_dict=state_dict,
                save_function=save_function,
                **kwargs,
            )
        dist.barrier()
        return None

    if merge_checkpoints:
        model_to_save = self.__class__(self.config).eval()
        tensor_parallel_engine = TensorParallelEngine(model_to_save, self.mpu, mapping)
        tensor_parallel_engine.parallelize()

        if state_dict is None:
            state_dict = unwrap_model(self).state_dict()

        model_to_save.load_state_dict(state_dict)

        if self.mpu.get_tensor_parallel_world_size() > 1:
            tensor_deparallel_engine = TensorDeparallelEngine(
                model_to_save, model_to_save.mpu, mapping
            )
            tensor_deparallel_engine.deparallelize()

        if dist.get_rank() == 0:
            model_to_save.save_pretrained(
                save_directory=save_directory,
                save_config=save_config,
                save_function=save_function,
                **kwargs,
            )

        del model_to_save

        dist.barrier()
        return None

    if os.path.isfile(save_directory):
        logger.error(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
        return

    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = unwrap_model(self)

    # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
    # we currently don't use this setting automatically, but may start to use with v5
    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.torch_dtype = str(dtype).split(".")[1]

    # Attach architecture to the config
    model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # Save the config
    if save_config:
        model_to_save.config.save_pretrained(save_directory)

    # Save the model
    if state_dict is None:
        state_dict = model_to_save.state_dict()

    # Handle the case where some state_dict keys shouldn't be saved
    if self._keys_to_ignore_on_save is not None:
        state_dict = {
            k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save
        }

    # If we save using the predefined names, we can load using `from_pretrained`
    weights_name = PARALLELIZED_WEIGHTS_NAME
    weights_name = weights_name.replace(
        "tp_0", f"tp_{self.mpu.get_tensor_parallel_rank()}"
    )
    weights_name = weights_name.replace(
        "pp_0", f"pp_{self.mpu.get_pipeline_parallel_rank()}"
    )

    output_model_file = os.path.join(save_directory, weights_name)

    if self.mpu.get_data_parallel_world_size() > 1:
        if self.mpu.get_data_parallel_rank() == 0:
            save_function(state_dict, output_model_file)
    else:
        save_function(state_dict, output_model_file)

    dist.barrier()
    logger.info(f"Model weights saved in {output_model_file}")
