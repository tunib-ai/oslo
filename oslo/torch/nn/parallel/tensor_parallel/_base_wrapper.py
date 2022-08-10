import copy
import os
import json

import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Union, Optional, Callable
from logging import getLogger

from oslo.torch.distributed import ParallelContext, ParallelMode

from oslo.torch.nn.parallel.utils import (
    ParallelWrapper,
    _update_module_arguments,
    is_huggingface_model,
    is_oslo_model,
    allocate_params,
    unwrap_parallel,
    get_parameter_dtype,
)


class BaseTensorParallelWrapper(ParallelWrapper):
    """
    PyTorch module for xD tensor parallelism

    Args:
        module (nn.Module): model object
        parallel_context (ParallelContext): parallel context object
    """

    def __init__(
        self,
        module: nn.Module,
        parallel_context: ParallelContext,
        mapping: dict = None,
        module_args: dict = None,
    ):
        super().__init__()

    @torch.no_grad()
    def save_parallelized(
        self,
        new_module,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        merge_checkpoints: bool = False,
        mapping: Optional[dict] = None,
        **kwargs,
    ):
        logger = getLogger("TensorParallel")
        PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"

        if (
            self.parallel_context.get_world_size(ParallelMode.TENSOR) == 1
            and self.parallel_context.get_world_size(ParallelMode.PIPELINE) == 1
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
            model_to_save = self.__class__(
                module=new_module,
                parallel_context=self.parallel_context,
                mapping=mapping,
                module_args=self.config,
            ).eval()

            if state_dict is None:
                state_dict = self.state_dict()

            model_to_save.load_state_dict(state_dict)
            allocate_params(model_to_save, self.parallel_context)

            if self.parallel_context.get_world_size(ParallelMode.TENSOR) > 1:
                model_to_save.deparallelize()

            if dist.get_rank() == 0:
                if is_huggingface_model(model_to_save.module):
                    model_to_save.module.save_pretrained(
                        save_directory=save_directory,
                        save_config=save_config,
                        save_function=save_function,
                        **kwargs,
                    )
                else:
                    if save_config:
                        with open(
                            os.path.join(save_directory, "config.json"), "w"
                        ) as f:
                            json.dump(self.config, f)
                    save_function(
                        model_to_save,
                        os.path.join(save_directory, "pytorch_model.bin"),
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
        model_to_save = unwrap_parallel(self)

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
        if getattr(self, "_keys_to_ignore_on_save", None) is not None:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in self._keys_to_ignore_on_save
            }

        # If we save using the predefined names, we can load using `from_pretrained`
        weights_name = PARALLELIZED_WEIGHTS_NAME
        weights_name = weights_name.replace(
            "tp_0", f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}"
        )
        weights_name = weights_name.replace(
            "pp_0", f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}"
        )

        output_model_file = os.path.join(save_directory, weights_name)

        if self.parallel_context.get_world_size(ParallelMode.DATA) > 1:
            if self.parallel_context.get_local_rank(ParallelMode.DATA) == 0:
                save_function(state_dict, output_model_file)
        else:
            save_function(state_dict, output_model_file)

        dist.barrier()
        logger.info(f"Model weights saved in {output_model_file}")

    def from_parallelized(self, path):
        """
        Example:
        >>> model = AnyModel()
        >>> model = TensorParallel(model, ...)
        >>> model.from_parallelized(path)
        """
        PARALLELIZED_WEIGHTS_NAME = "pytorch_model_tp_0_pp_0.bin"
        parallelized_model_path = path

        file_names = {
            os.path.join(
                parallelized_model_path,
                PARALLELIZED_WEIGHTS_NAME.replace("tp_0", f"tp_{tp}").replace(
                    "pp_0", f"pp_{pp}"
                ),
            )
            for tp in range(self.parallel_context.get_world_size(ParallelMode.TENSOR))
            for pp in range(self.parallel_context.get_world_size(ParallelMode.PIPELINE))
        }

        if os.path.isdir(parallelized_model_path):
            if all(os.path.isfile(file_name) for file_name in file_names):
                state_dict = torch.load(
                    os.path.join(
                        parallelized_model_path,
                        PARALLELIZED_WEIGHTS_NAME.replace(
                            "tp_0",
                            f"tp_{self.parallel_context.get_local_rank(ParallelMode.TENSOR)}",
                        ).replace(
                            "pp_0",
                            f"pp_{self.parallel_context.get_local_rank(ParallelMode.PIPELINE)}",
                        ),
                    )
                )

                if getattr(self, "_keys_to_ignore_on_save", None) is not None:
                    state_dict = {
                        k: v
                        for k, v in state_dict.items()
                        if k not in self._keys_to_ignore_on_save
                    }

                self.load_state_dict(state_dict=state_dict, strict=False)

            else:
                raise FileNotFoundError(
                    f"all the {file_names} are necessary. "
                    f"but some of them do not exist. Please check your checkpoint files."
                )
        else:
            raise NotADirectoryError(
                f"directory named {parallelized_model_path} is not valid. "
            )

    @torch.no_grad()
    def deparallelize(self):
        return NotImplementedError
