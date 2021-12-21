# Copyright 2021 TUNiB Inc.

import logging
import os
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
from transformers import cached_path
from transformers.modeling_utils import get_parameter_dtype, unwrap_model

from oslo.parallelism.mpu import MPU
from oslo.parallelism.pipeline_parallelism import (
    PipelineDeparallelEngine,
    PipelineParallelEngine,
)
from oslo.parallelism.tensor_parallelism import (
    TensorDeparallelEngine,
    TensorParallelEngine,
)

logger = logging.getLogger(__name__)
WEIGHTS_NAME_PARALLEL = "pytorch_model_tp=0_pp=0.bin"


class ParallelizationMixin(object):
    """
    ParallelizationMixin is the module that
    provides model parallelism capabilities to the transformers model.

    Notes:
        we provide the following methods: ``xxxx_with_parallel(...)``

        - ``from_pretrained(...)`` -> ``from_pretrained_with_parallel(...)``
        - ``save_pretrained(...)`` -> ``save_pretrained_with_parallel(...)``

    Examples:
        >>> # parallelized pre-trained model with checkpoint in hub.
        >>> model = GPT2LMHeadModel.from_pretrained_with_parallel("gpt2", ...)

        >>> # parallelized pre-trained model with checkpoint in local machine.
        >>> # both merged form (pytorch_model.bin) and split form (pytorch_model_tp=0_pp=0.bin) are supported.
        >>> model = GPT2LMHeadModel.from_pretrained_with_parallel("path/to/model", ...)

        >>> # parallelized random-initialized model with config
        >>> model = GPT2LMHeadModel.from_config_with_parallel(config, ...)

        >>> # save trained model in split form
        >>> model.saved_pretrained_with_parallel(save_with_merge=False, ...)
        model checkpoint will be saved like 'pytorch_model_tp=0_pp=0.bin'

        >>> # save trained model in merged form (It requires more memory usage)
        >>> model.saved_pretrained_with_parallel(save_with_merge=True, ...)
        model checkpoint will be saved like 'pytorch_model.bin'
    """

    # Every models in OSLO must have the following three methods for model parallelism.
    # These methods provide basic information of layers in model.

    @staticmethod
    def get_layer_policies():
        """Return list of layer policies"""
        return []

    def get_head_layers(self):
        """Return list of head layers"""
        return []

    # We split ``forward`` method into 6 sub-methods for pipeline parallelism.
    # Previously, the existing libraries such as DeepSpeed and TorchGPipe split the model's layers and created the ``nn.Sequential``.
    # Instead of splitting the layers, we split the methods to enable pipeline parallelism.

    def preblock_fn(self, *args, **kwargs):
        """Forward before main block of model"""
        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.preblock_fn(*args, **kwargs)

    def block_fn(self, *args, **kwargs):
        """Forward of main block that is pipeline parallelized"""
        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.block_fn(*args, **kwargs)

    def postblock_fn(self, *args, **kwargs):
        """Forward after main block of model"""
        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.postblock_fn(*args, **kwargs)

    def head_fn(self, *args, **kwargs):
        """Forward Head after postblock function"""
        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.head_fn(*args, **kwargs)

    def loss_fn(self, *args, **kwargs):
        """Model and task specific loss function"""
        if kwargs.get("labels", None) is None:
            return None

        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.loss_fn(*args, **kwargs)

    def organize_fn(self, *args, **kwargs):
        """Organize outputs of ``forward_postblock`` to provide user"""
        base_model = self.base_model

        if base_model is self:
            return kwargs
        else:
            return base_model.organize_fn(*args, **kwargs)

    # Utility methods

    @staticmethod
    def make_outputs(kwargs, **outputs):
        kwargs.update(outputs)
        return kwargs

    def is_model_parallelized(self) -> bool:
        """Whether model is parallelized or not"""
        return hasattr(self, "mpu")

    def is_data_parallelized(self) -> bool:
        """Whether model is parallelized or not"""
        return (
            self.is_model_parallelized() and self.mpu.get_data_parallel_world_size() > 1
        )

    def is_tensor_parallelized(self) -> bool:
        """Whether model is tensor parallelized or not"""
        return (
            self.is_model_parallelized()
            and self.mpu.get_tensor_parallel_world_size() > 1
        )

    def is_pipeline_parallelized(self) -> bool:
        """Whether model is pipeline parallelized or not"""
        return (
            self.is_model_parallelized()
            and self.mpu.get_pipeline_parallel_world_size() > 1
        )

    # These methods parallelize model via parallelization engines.

    def _tensor_parallelize(self, mpu: MPU):
        """
        Conduct tensor model parallelization

        Args:
            mpu (MPU): model parallel unit
        """
        for policy in self.get_layer_policies():
            tpe = TensorParallelEngine(
                mpu=mpu,
                policy=policy,
                head_layers=self.get_head_layers(),
            )

            tpe.parallelize(model=self)

    def _pipeline_parallelize(self, mpu: MPU, micro_batch_size: int):
        """
        Conduct pipeline model parallelization

        Args:
            mpu (MPU): model parallel unit
            micro_batch_size (int): micro batch size
        """
        for policy in self.get_layer_policies():
            ppe = PipelineParallelEngine(
                mpu=mpu,
                policy=policy,
                head_layers=self.get_head_layers(),
                micro_batch_size=micro_batch_size,
            )

            ppe.parallelize(model=self)

    # Save methods with auto merging mechanism
    @torch.no_grad()
    def save_pretrained_with_parallel(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        save_with_merging: bool = False,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        **kwargs,
    ):
        if not self.is_tensor_parallelized() and not self.is_pipeline_parallelized():
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

        if save_with_merging:
            model_to_save = self.__class__.from_config_with_parallel(
                self.config,
                **self.initial_parameters,
            ).eval()

            if state_dict is None:
                state_dict = unwrap_model(self).state_dict()

            model_to_save.load_state_dict(state_dict)

            if self.is_tensor_parallelized():
                for policy in model_to_save.get_layer_policies():
                    tde = TensorDeparallelEngine(
                        mpu=model_to_save.mpu,
                        policy=policy,
                        head_layers=model_to_save.get_head_layers(),
                    )

                    tde.deparallelize(model_to_save)

            if self.is_pipeline_parallelized():
                for policy in model_to_save.get_layer_policies():
                    pde = PipelineDeparallelEngine(
                        mpu=model_to_save.mpu,
                        policy=policy,
                        head_layers=model_to_save.get_head_layers(),
                    )

                    pde.deparallelize(model_to_save)

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
                k: v
                for k, v in state_dict.items()
                if k not in self._keys_to_ignore_on_save
            }

        # If we save using the predefined names, we can load using `from_pretrained`
        weights_name = WEIGHTS_NAME_PARALLEL
        weights_name = weights_name.replace(
            "tp=0", f"tp={self.mpu.get_tensor_parallel_rank()}"
        )
        weights_name = weights_name.replace(
            "pp=0", f"pp={self.mpu.get_pipeline_parallel_rank()}"
        )

        output_model_file = os.path.join(save_directory, weights_name)

        if self.is_data_parallelized():
            if self.mpu.get_data_parallel_rank() == 0:
                save_function(state_dict, output_model_file)
        else:
            save_function(state_dict, output_model_file)

        dist.barrier()
        logger.info(f"Model weights saved in {output_model_file}")

    def gpu_modules(self):
        """For DDP or DeepSpeed ZeRO Data parallelism"""
        return getattr(self.__class__, "pipe_modules", self)

    def gpu_parameters(self):
        """For DDP or DeepSpeed ZeRO Data parallelism"""
        return self.gpu_modules().parameters()

    @classmethod
    def from_pretrained_with_parallel(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        pipeline_parallel_size = kwargs.pop("pipeline_parallel_size", 1)
        micro_batch_size = kwargs.pop("micro_batch_size", 1)
        resize_token_embeddings = kwargs.pop("resize_token_embeddings", None)

        assert (
            cls.is_parallelizable is True
        ), "This model doesn't support tensor model parallelism. please check the document."

        assert (
            tensor_parallel_size >= 1
        ), "param `tensor_parallel_size` must be positive."
        assert (
            tensor_parallel_size & (tensor_parallel_size - 1) == 0
        ), "param `tensor_parallel_size` must be power of 2."
        assert (
            pipeline_parallel_size >= 1
        ), "param `pipeline_parallel_size` must be positive."

        if os.path.isdir(pretrained_model_name_or_path):
            # segmented checkpoints
            # - pytorch_model_tp=0_pp=0.bin
            # - pytorch_model_tp=1_pp=0.bin
            # - pytorch_model_tp=2_pp=0.bin
            # - pytorch_model_tp=3_pp=0.bin
            # ...

            file_names = {
                os.path.join(
                    pretrained_model_name_or_path,
                    WEIGHTS_NAME_PARALLEL.replace("tp=0", f"tp={tp}").replace(
                        "pp=0", f"pp={pp}"
                    ),
                )
                for tp in range(tensor_parallel_size)
                for pp in range(pipeline_parallel_size)
            }

            if all(os.path.isfile(file_name) for file_name in file_names):
                cache_dir = kwargs.pop("cache_dir", None)
                force_download = kwargs.pop("force_download", False)
                resume_download = kwargs.pop("resume_download", False)
                proxies = kwargs.pop("proxies", None)
                local_files_only = kwargs.pop("local_files_only", False)
                use_auth_token = kwargs.pop("use_auth_token", None)
                revision = kwargs.pop("revision", None)
                from_pipeline = kwargs.pop("_from_pipeline", None)
                from_auto_class = kwargs.pop("_from_auto", False)
                _fast_init = kwargs.pop("_fast_init", True)
                output_loading_info = kwargs.pop("output_loading_info", False)
                low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
                ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)

                config, model_kwargs = cls.config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    cache_dir=cache_dir,
                    return_unused_kwargs=True,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )

                model = cls.from_config_with_parallel(
                    config,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    micro_batch_size=micro_batch_size,
                    resize_token_embeddings=resize_token_embeddings,
                    *model_args,
                    **kwargs,
                )

                tensor_parallel_rank = model.mpu.get_tensor_parallel_rank()
                pipeline_parallel_rank = model.mpu.get_pipeline_parallel_rank()

                archive_file = WEIGHTS_NAME_PARALLEL
                archive_file = archive_file.replace(
                    "tp=0", f"tp={tensor_parallel_rank}"
                )
                archive_file = archive_file.replace(
                    "pp=0", f"pp={pipeline_parallel_rank}"
                )
                archive_file = os.path.join(pretrained_model_name_or_path, archive_file)

                try:
                    # Load from URL or cache if already cached
                    user_agent = {
                        "file_type": "model",
                        "framework": "pytorch",
                        "from_auto_class": from_auto_class,
                    }

                    resolved_archive_file = cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME_PARALLEL}\n\n"
                    )

                    if revision is not None:
                        msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"

                    raise EnvironmentError(msg)

                state_dict = torch.load(resolved_archive_file, map_location="cpu")

                if low_cpu_mem_usage:
                    # save the keys
                    loaded_state_dict_keys = [k for k in state_dict.keys()]
                    del state_dict  # free CPU memory - will reload again later
                    cls._load_state_dict_into_model_low_mem(
                        model, loaded_state_dict_keys, resolved_archive_file
                    )

                else:
                    (
                        model,
                        missing_keys,
                        unexpected_keys,
                        mismatched_keys,
                        error_msgs,
                    ) = cls._load_state_dict_into_model(
                        model,
                        state_dict,
                        pretrained_model_name_or_path,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        _fast_init=_fast_init,
                    )

                # make sure token embedding weights are still tied if needed
                model.tie_weights()

                # Set model in evaluation mode to deactivate DropOut modules by default
                model.eval()

                setattr(model, "initial_parameters", {})
                model.initial_parameters["tensor_parallel_size"] = tensor_parallel_size
                model.initial_parameters[
                    "pipeline_parallel_size"
                ] = pipeline_parallel_size
                model.initial_parameters[
                    "resize_token_embeddings"
                ] = resize_token_embeddings
                model.initial_parameters.update(**kwargs)

                if output_loading_info:
                    loading_info = {
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                        "mismatched_keys": mismatched_keys,
                        "error_msgs": error_msgs,
                    }
                    return model, loading_info

                return model
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME_PARALLEL)
            ):
                raise EnvironmentError(
                    f"files named {file_names} are necessary. "
                    f"but some files do not exist. Please check your checkpoint files."
                )

        # 1. model name on hub (e.g. gpt2, bert-base-cased, ...)
        # 2. merged checkpoint (pytorch_model.bin)
        model = cls.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        if resize_token_embeddings is not None:
            model.resize_token_embeddings(resize_token_embeddings)

        mpu = MPU(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        if pipeline_parallel_size > 1:
            model._pipeline_parallelize(mpu=mpu, micro_batch_size=micro_batch_size)

        if tensor_parallel_size > 1:
            model._tensor_parallelize(mpu=mpu)

        setattr(model, "initial_parameters", {})
        model.initial_parameters["tensor_parallel_size"] = tensor_parallel_size
        model.initial_parameters["pipeline_parallel_size"] = pipeline_parallel_size
        model.initial_parameters["micro_batch_size"] = micro_batch_size
        model.initial_parameters["resize_token_embeddings"] = resize_token_embeddings
        model.initial_parameters.update(**kwargs)

        setattr(model, "mpu", mpu)
        setattr(model.base_model, "mpu", mpu)
        # This allows the Trainer to call the MPU for data + model parallelism
        # example `ddp = DistributedDataParallel(..., process_group=model.mpu.get_data_parallel_group())`

        return model

    @classmethod
    def from_config_with_parallel(cls, config, *model_args, **kwargs):
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        pipeline_parallel_size = kwargs.pop("pipeline_parallel_size", 1)
        micro_batch_size = kwargs.pop("micro_batch_size", 1)
        resize_token_embeddings = kwargs.pop("resize_token_embeddings", None)

        assert (
            cls.is_parallelizable is True
        ), "This model doesn't support tensor model parallelism. please check the document."
        assert (
            tensor_parallel_size >= 1
        ), "param `tensor_parallel_size` must be positive."
        assert (
            tensor_parallel_size & (tensor_parallel_size - 1) == 0
        ), "param `tensor_parallel_size` must be power of 2."
        assert (
            pipeline_parallel_size >= 1
        ), "param `pipeline_parallel_size` must be positive."

        for k, v in kwargs.items():
            setattr(config, k, v)

        model = cls._from_config(config)

        if resize_token_embeddings is not None:
            model.resize_token_embeddings(resize_token_embeddings)

        mpu = MPU(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        if pipeline_parallel_size > 1:
            model._pipeline_parallelize(mpu=mpu, micro_batch_size=micro_batch_size)

        if tensor_parallel_size > 1:
            model._tensor_parallelize(mpu=mpu)

        setattr(model, "initial_parameters", {})
        model.initial_parameters["tensor_parallel_size"] = tensor_parallel_size
        model.initial_parameters["pipeline_parallel_size"] = pipeline_parallel_size
        model.initial_parameters["micro_batch_size"] = micro_batch_size
        model.initial_parameters["resize_token_embeddings"] = resize_token_embeddings
        model.initial_parameters.update(**kwargs)

        setattr(model, "mpu", mpu)
        setattr(model.base_model, "mpu", mpu)
        # This allows the Trainer to call the MPU for data + model parallelism
        # example `ddp = DistributedDataParallel(..., process_group=model.mpu.get_data_parallel_group())`

        return model
