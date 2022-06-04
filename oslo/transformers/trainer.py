import contextlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm.auto import tqdm
import datasets
import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .training_args import ParallelMode, TrainingArguments
from .trainer_utils import (
    set_seed,
    has_length,
    EvalPrediction,
    get_last_checkpoint,
    TrainOutput,
    ShardedDDPOption,
    speed_metrics,
    unwrap_model,
    PredictionOutput,
    EvalLoopOutput,
    denumpify_detensorize,
    TrainerMemoryTracker,
)
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    # PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    LabelSmoother,
    IterableDatasetShard,
    ShardSampler,
    SequentialDistributedSampler,
    get_parameter_names,
    LengthGroupedSampler,
    nested_numpify,
    find_batch_size,
    nested_concat,
    nested_truncate,
    nested_detach,
    distributed_concat,
    DistributedLengthGroupedSampler,
    distributed_broadcast_scalars,
)
from .integrations import (  # isort: split
    get_reporting_integration_callbacks)
from .utils import (
    logging,
    find_labels,
)

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if TYPE_CHECKING:
    import optuna

PREFIX_CHECKPOINT_DIR = "checkpoint"
logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.yaml"

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class Trainer:

    def __init__(
        self,
        model: nn.Module = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor],
                                                torch.Tensor] = None,
    ):
        self.deepspeed = None  # temp
        if args is None:
            # No Arguments passed
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        set_seed(self.args.seed)
        self.hp_name = None
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(
            self.args.skip_memory_metrics)
        self._memory_tracker.start()

        self.is_in_train = False

        # set log level
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        # # Remove model_init
        # if model is None:
        #     if model_init is not None:
        #         self.model_init = model_init
        #         model = self.call_model_init()
        # else:
        #     if model_init is not None:
        #         warnings.warn(
        #             "`Trainer` requires either a `model` or `model_init` argument, but not both. "
        #             "`model_init` will overwrite your model when calling the `train` method. This will become a fatal error in the next release.",
        #             FutureWarning,
        #         )
        #     self.model_init = model_init

        if hasattr(model, "is_parallelizable"
                  ) and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # TODO - Task C Sharded_ddp

        self.sharded_ddp = None
        """
        if len(args.sharded_ddp) > 0:
            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif ShardedDDPOption.SIMPLE not in args.sharded_ddp and FullyShardedDDP is None:
                raise ImportError(
                    "Sharded DDP in a mode other than simple training requires fairscale version >= 0.3, found "
                    f"{fairscale.__version__}. Upgrade your fairscale library: `pip install --upgrade fairscale`."
                )
            elif ShardedDDPOption.SIMPLE in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.SIMPLE
            elif ShardedDDPOption.ZERO_DP_2 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_2
            elif ShardedDDPOption.ZERO_DP_3 in args.sharded_ddp:
                self.sharded_ddp = ShardedDDPOption.ZERO_DP_3
        """

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. Sharded DDP - same as MP
        self.place_model_on_device = args.place_model_on_device  # GPU에 올릴 것인지 말건지
        if (self.is_model_parallel
                # or args.deepspeed
                or
            ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
                # TODO; or (self.sharded_ddp in [ShardedDDPOption.ZERO_DP_2, ShardedDDPOption.ZERO_DP_3])
           ):
            self.place_model_on_device = False

        # TODO - Task D: Create src/transformers/data related to Trainer
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(
            tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # TODO - Task E: Create self._move_model_to_device
        # if self.place_model_on_device:
        #     self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers

        # if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
        #     raise RuntimeError(
        #         "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
        #         "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
        #     )

        # Create all CALLBACKS functions (includes TrainerCallback)
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(callbacks, self.model,
                                                self.tokenizer, self.optimizer,
                                                self.lr_scheduler)
        # self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.callback_handler.add_callback(DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # remove push to hub

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(
                getattr(self.data_collator, "collate_batch", None)):
            raise ValueError(
                "The `data_collator` should be a simple callable (function, class with `__call__`)."
            )

        if args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if train_dataset is not None and not has_length(
                train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )

        # remove group_by_length

        self._signature_columns = None

        # Mixed precision setup
        self.use_amp = False
        if args.fp16 or args.bf16:
            if args.half_precision_backend == "auto":
                if _is_native_amp_available:
                    args.half_precision_backend = "amp"
                else:
                    if args.bf16:
                        raise ValueError(
                            "Tried to use `bf16` but native amp is not available"
                        )
                    raise ValueError(
                        "Tried to use `fp16` but native amp is not available")
            logger.info(
                f"Using {args.half_precision_backend} half precision backend")

        self.do_grad_scaling = False
        if (args.fp16 or args.bf16
           ) and not args.deepspeed:  # deepspeed manages its own half precision
            if args.half_precision_backend == "amp":
                self.use_amp = True
                self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
                self.do_grad_scaling = True
                if self.sharded_ddp is not None:
                    # TODO change sharded ddp to oslo's
                    # self.scaler = ShardedGradScaler()
                    self.scaler = torch.cuda.amp.GradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()
            else:
                raise ImportError(
                    "Using FP16 with torch.amp but it is not installed")

        # Set LabelSmoother
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        # Set TrainerState
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()

        # TODO check should we support count flos ?
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False  # ?

        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # TODO memory_tracker
        args = self.args
        self.is_in_train = True

        # TODO move model to device

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        # if resume_from_checkpoint is not None:
        #     self._load_from_checkpoint(resume_from_checkpoint)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0)
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs *
                                      num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        delay_optimizer_creation = (self.sharded_ddp is not None and
                                    self.sharded_ddp != ShardedDDPOption.SIMPLE)

        # TODO Debugoption
        # TODO Shareded ddp engine, optimizer, lr_sheduler

        # if args.deepspeed:
        #     deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
        #         self,
        #         num_training_steps=max_steps,
        #         resume_from_checkpoint=resume_from_checkpoint)
        #     self.model = deepspeed_engine.module
        #     self.model_wrapped = deepspeed_engine
        #     self.deepspeed = deepspeed_engine
        #     self.optimizer = optimizer
        #     self.lr_scheduler = lr_scheduler

        # TODO: delay_optimizer_creation
        # elif not delay_optimizer_creation:
        #     self.create_optimizer()
        #     self.create_scheduler(num_training_steps=max_steps,
        #                           optimizer=self.optimizer)
        # shared ddp option

        self.state = TrainerState()
        # self.state.is_hyper_param_search = trial is not None

        # TODO gradient checkpointing ??
        # for the rest of this function `model` is the outside model, whether it was wrapped or not

        model = self._wrap_model(self.model_wrapped)
        if model is not self.model:
            self.model_wrapped = model

        # TODO: delay_optimizer_creation
        # if delay_optimizer_creation:
        #     self.create_optimizer()
        #     self.create_scheduler(num_training_steps=max_steps,
        #                           optimizer=self.optimizer)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero():
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(
            trial) if self.hp_name is not None else None

        # if trial is not None:
        #     assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
        #     self.state.trial_params = hp_params(assignments)
        # else:
        #     self.state.trial_params = None

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)

        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        # set callback handler
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        # TODO ?? # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        # if not args.ignore_data_skip:
        #     for epoch in range(epochs_trained):
        #         is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
        #             train_dataloader.sampler, RandomSampler
        #         )
        #         if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
        #             # We just need to begin an iteration to create the randomization of the sampler.
        #             # That was before PyTorch 1.11 however...
        #             for _ in train_dataloader:
        #                 break
        #         else:
        #             # Otherwise we need to call the whooooole sampler cause there is some random operation added
        #             # AT THE VERY END!
        #             _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                    train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (len(epoch_iterator) if len_dataloader is not None
                              else args.max_steps *
                              args.gradient_accumulation_steps)
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            # step = -1
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                # Start step begin
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                if (((step + 1) % args.gradient_accumulation_steps != 0) and
                        args.local_rank != -1 and
                        args._no_sync_in_gradient_accumulation):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (args.logging_nan_inf_filter and
                    (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step -
                                          self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                # TODO support calculate count flos per each process
                # self.current_flos += float(self.floating_point_ops(inputs))

                # ddp
                # optimizing ddp
                if (step + 1
                   ) % args.gradient_accumulation_steps == 0 or (
                       # last step in epoch but step is always smaller than gradient_accumulation_steps
                       steps_in_epoch <= args.gradient_accumulation_steps and
                       (step + 1) == steps_in_epoch):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        # else:
                        #     # Revert to normal clipping otherwise, handling Apex or full precision
                        #     nn.utils.clip_grad_norm_(
                        #         amp.master_params(self.optimizer)
                        #         if self.use_apex else model.parameters(),
                        #         args.max_grad_norm,
                        #     )

                    # optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
                                                  ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if step < 0:
                    logger.warning(
                        f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(
                    args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
                                              ignore_keys_for_eval)

                if self.control.should_training_stop:
                    break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train",
                                start_time,
                                num_samples=num_train_samples,
                                num_steps=self.state.max_steps)
        # TODO self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        # TODO _memory_tracker
        # self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            ))

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(self,
                test_dataset: Dataset,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "test") -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(test_dataloader,
                                      description="Prediction",
                                      ignore_keys=ignore_keys,
                                      metric_key_prefix=metric_key_prefix)
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            ))

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions,
                                label_ids=output.label_ids,
                                metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self,
                                                    num_training_steps=0,
                                                    resume_from_checkpoint=None,
                                                    inference=True)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model,
                                                        inputs,
                                                        prediction_loss_only,
                                                        ignore_keys=ignore_keys)
            inputs_decode = inputs[
                "input_ids"] if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode if inputs_host is None else nested_concat(
                        inputs_host, inputs_decode, padding_index=-100))
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (
                    step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode if all_inputs is None else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100))
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (labels
                                  if all_labels is None else nested_concat(
                                      all_labels, labels, padding_index=-100))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (inputs_decode
                          if all_inputs is None else nested_concat(
                              all_inputs, inputs_decode, padding_index=-100))
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(
                eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds,
                                   label_ids=all_labels,
                                   inputs=all_inputs))
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds,
                              label_ids=all_labels,
                              metrics=metrics,
                              num_samples=num_samples)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if self.args.local_rank != -1:
            tensors = distributed_concat(tensors)
        return tensors

    # Copied from Accelerate.
    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(
                self._pad_across_processes(t, pad_index=pad_index)
                for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({
                k: self._pad_across_processes(v, pad_index=pad_index)
                for k, v in tensor.items()
            })
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        if tensor.shape[1] == max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, :old_size[1]] = tensor
        return new_tensor

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config,
                                      "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(
                tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model,
                                                      inputs,
                                                      return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items()
                                   if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset,
                                                       description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        """
        if shared ddp:
           wrap optimizer 
        """
        return self.optimizer

    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.
    #
    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #     """
    #     if self.lr_scheduler is None:
    #         self.lr_scheduler = get_scheduler(
    #             self.args.lr_scheduler_type,
    #             optimizer=self.optimizer if optimizer is None else optimizer,
    #             num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #             num_training_steps=num_training_steps,
    #         )
    #     return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError
               ):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    # def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
    #     """HP search setup code"""
    #     self._trial = trial
    #
    #     if self.hp_search_backend is None or trial is None:
    #         return
    #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
    #         params = self.hp_space(trial)
    #     elif self.hp_search_backend == HPSearchBackend.RAY:
    #         params = trial
    #         params.pop("wandb", None)
    #     elif self.hp_search_backend == HPSearchBackend.SIGOPT:
    #         params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}
    #     elif self.hp_search_backend == HPSearchBackend.WANDB:
    #         params = trial
    #
    #     for key, value in params.items():
    #         if not hasattr(self.args, key):
    #             logger.warning(
    #                 f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
    #             )
    #             continue
    #         old_attr = getattr(self.args, key, None)
    #         # Casting value to the proper type
    #         if old_attr is not None:
    #             value = type(old_attr)(value)
    #         setattr(self.args, key, value)
    #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
    #         logger.info("Trial:", trial.params)
    #     if self.hp_search_backend == HPSearchBackend.SIGOPT:
    #         logger.info(f"SigOpt Assignments: {trial.assignments}")
    #     if self.hp_search_backend == HPSearchBackend.WANDB:
    #         logger.info(f"W&B Sweep parameters: {trial}")
    #     if self.args.deepspeed:
    #         # Rebuild the deepspeed config to reflect the updated training parameters
    #         from transformers.deepspeed import HfTrainerDeepSpeedConfig
    #
    #         self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
    #         self.args.hf_deepspeed_config.trainer_config_process(self.args)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch,
                                 ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar /
                (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # self._report_to_hp_search(trial, epoch, metrics) TODO hp search

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        local_rank = self.args.local_rank
        if local_rank != -1:
            rng_file = os.path.join(checkpoint, f"rng_state_{local_rank}.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    f"Didn't find an RNG file for process {local_rank}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed.")
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.local_rank != -1:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state_all(
                        checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # TODO Support hp search func later.
        # if self.hp_search_backend is not None and trial is not None:
        #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
        #         run_id = trial.number
        #     elif self.hp_search_backend == HPSearchBackend.RAY:
        #         from ray import tune
        #
        #         run_id = tune.get_trial_id()
        #     elif self.hp_search_backend == HPSearchBackend.SIGOPT:
        #         run_id = trial.id
        #     elif self.hp_search_backend == HPSearchBackend.WANDB:
        #         import wandb
        #
        #         run_id = wandb.run.id
        #     run_name = self.hp_name(
        #         trial) if self.hp_name is not None else f"run-{run_id}"
        #     run_dir = os.path.join(self.args.output_dir, run_name)
        # else:
        #     run_dir = self.args.output_dir
        #     self.store_flos()
        run_dir = self.args.output_dir
        self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(),
                       os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(),
                           os.path.join(output_dir, SCHEDULER_NAME))
            # reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(),
                           os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or
                    self.state.best_model_checkpoint is None or
                    operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir,
                                                 TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states,
                       os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.deepspeed:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            return

        if os.path.isfile(os.path.join(
                checkpoint, OPTIMIZER_NAME)) and os.path.isfile(
                    os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states

            map_location = self.args.device
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, OPTIMIZER_NAME),
                           map_location=map_location))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            # reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling and os.path.isfile(
                    os.path.join(checkpoint, SCALER_NAME)):
                self.scaler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCALER_NAME)))

    def _get_learning_rate(self):
        if self.deepspeed:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning(
                        "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                    )
                    last_lr = 0
                else:
                    raise
        else:
            last_lr = (
                # backward compatibility for pytorch schedulers
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4") else
                self.lr_scheduler.get_lr()[0])
        return last_lr

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(
                    load_result.missing_keys) == set(
                        self.model._keys_to_ignore_on_save):
                self.model.tie_weights()
            else:
                logger.warning(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _load_best_model(self):
        logger.info(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )

        best_model_path = os.path.join(self.state.best_model_checkpoint,
                                       WEIGHTS_NAME)
        if os.path.exists(best_model_path):
            # TODO for shared ddp
            # if self.deepspeed:
            #     # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
            #     deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
            #     self.model = deepspeed_engine.module
            #     self.model_wrapped = deepspeed_engine
            #     self.deepspeed = deepspeed_engine
            #     self.optimizer = optimizer
            #     self.lr_scheduler = lr_scheduler
            #     self.deepspeed.load_checkpoint(
            #         self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            #     )
            # else:

            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(best_model_path, map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`.")

    # def _load_from_checkpoint(self, resume_from_checkpoint):
    #     if not os.path.isfile(os.path.join(resume_from_checkpoint,
    #                                        WEIGHTS_NAME)):
    #         raise ValueError(
    #             f"Can't find a valid checkpoint at {resume_from_checkpoint}")
    #
    #     logger.info(f"Loading model from {resume_from_checkpoint}).")
    #
    #     if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
    #         config = PretrainedConfig.from_json_file(
    #             os.path.join(resume_from_checkpoint, CONFIG_NAME))
    #         checkpoint_version = config.transformers_version
    #         if checkpoint_version is not None and checkpoint_version != __version__:
    #             logger.warning(
    #                 f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
    #                 f"Transformers but your current version is {__version__}. This is not recommended and could "
    #                 "yield to errors or unwanted behaviors.")
    #
    #     if self.args.deepspeed:
    #         # will be resumed in deepspeed_init
    #         pass
    #     else:
    #         # We load the model state dict on the CPU to avoid an OOM error.
    #         state_dict = torch.load(os.path.join(resume_from_checkpoint,
    #                                              WEIGHTS_NAME),
    #                                 map_location="cpu")
    #         # If the model is on the GPU, it still works!
    #         self._load_state_dict_in_model(state_dict)
    #
    #         # release memory
    #         del state_dict

    # def call_model_init(self, trial=None):
    #     model_init_argcount = number_of_arguments(self.model_init)
    #     if model_init_argcount == 0:
    #         model = self.model_init()
    #     elif model_init_argcount == 1:
    #         model = self.model_init(trial)
    #     else:
    #         raise RuntimeError("model_init should have 0 or 1 argument.")
    #
    #     if model is None:
    #         raise RuntimeError("model_init should not return None.")
    #
    #     return model
    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(
                model, "tie_weights"):
            model.tie_weights()

    def _remove_unused_columns(self,
                               dataset: "datasets.Dataset",
                               description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]

        ignored_columns = list(
            set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message.")

        columns = [
            k for k in self._signature_columns if k in dataset.column_names
        ]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(type=dataset.format["type"],
                               columns=columns,
                               format_kwargs=dataset.format["format_kwargs"])
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if isinstance(self.train_dataset, datasets.Dataset):
                lengths = (self.train_dataset[self.args.length_column_name]
                           if self.args.length_column_name
                           in self.train_dataset.column_names else None)
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[
                0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size *
                    self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size *
                    self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset,
                                         generator=generator)
                return RandomSampler(self.train_dataset)
            # elif (self.args.parallel_mode in [
            #         ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL
            # ] and not self.args.dataloader_drop_last):
            #     # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            #     return DistributedSamplerWithLoop(
            #         self.train_dataset,
            #         batch_size=self.args.per_device_train_batch_size,
            #         num_replicas=self.args.world_size,
            #         rank=self.args.process_index,
            #         seed=seed,
            #     )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset,
                                                        description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_eval_sampler(
            self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

    def training_step(
            self, model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean(
            )  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        # TODO
        # elif self.deepspeed:
        #     # loss gets scaled under gradient_accumulation_steps in deepspeed
        #     loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def _prepare_input(
            self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)(
                {k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            # if self.deepspeed and data.dtype != torch.int64:
            #     # NLP models inputs are int64 and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        ctx_manager = contextlib.nullcontext() if sys.version_info >= (
            3, 7) else contextlib.suppress()

        return ctx_manager

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _wrap_model(self, model, training=True):

        # # already initialized its own DDP and AMP
        # if self.deepspeed:
        #     return self.deepspeed

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model
        #
        # # Mixed precision training with apex (torch < 1.6)
        # if self.use_apex and training:
        #     model, self.optimizer = amp.initialize(
        #         model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # TODO Distributed training (should be after apex fp16 initialization)
        # if self.sharded_ddp is not None:
        #     # Sharded DDP!
        #     if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #         model = ShardedDDP(model, self.optimizer)
        #     else:
        #         mixed_precision = self.args.fp16 or self.args.bf16
        #         cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
        #         zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
        #         # XXX: Breaking the self.model convention but I see no way around it for now.
        #         if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
        #             model = auto_wrap(model)
        #         self.model = model = FullyShardedDDP(
        #             model,
        #             mixed_precision=mixed_precision,
        #             reshard_after_forward=zero_3,
        #             cpu_offload=cpu_offload,
        #         ).to(self.args.device)

        elif self.args.local_rank != -1:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs[
                    "find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs[
                    "find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank]
                if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank
                if self.args._n_gpu != 0 else None,
                **kwargs,
            )

        return model

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state,
                                                    self.control, logs)

    def save_model(self,
                   output_dir: Optional[str] = None,
                   _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if (ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp or
                ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp):
            state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        # elif self.deepspeed:
        #
        #     # this takes care of everything as long as we aren't under zero3
        #     if self.args.should_save:
        #         self._save(output_dir)
        #
        #     if is_deepspeed_zero3_enabled():
        #         # It's too complicated to try to override different places where the weights dump gets
        #         # saved, so since under zero3 the file is bogus, simply delete it. The user should
        #         # either user deepspeed checkpoint to resume or to recover full weights use
        #         # zero_to_fp32.py stored in the checkpoint.
        #         if self.args.should_save:
        #             file = os.path.join(output_dir, WEIGHTS_NAME)
        #             if os.path.isfile(file):
        #                 # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
        #                 os.remove(file)
        #
        #         # now save the real model if stage3_gather_16bit_weights_on_model_save=True
        #         # if false it will not be saved.
        #         # This must be called on all ranks
        #         if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
        #             logger.warning(
        #                 "deepspeed.save_16bit_model didn't save the model, since stage3_gather_16bit_weights_on_model_save=false. "
        #                 "Saving the full checkpoint instead, use zero_to_fp32.py to recover weights"
        #             )
        #             self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir,
                                                         state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.args.local_rank != -1:
            self.state.total_flos += (distributed_broadcast_scalars(
                [self.current_flos], device=self.args.device).sum().item())
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _sorted_checkpoints(self,
                            output_dir=None,
                            checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
                            use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x)
            for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
            if os.path.isdir(x)
        ]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [
            checkpoint[1] for checkpoint in checkpoints_sorted
        ]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(
                str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[
                    i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime,
                                                      output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (self.state.best_model_checkpoint is not None and
                self.args.save_total_limit == 1 and
                checkpoints_sorted[-1] != self.state.best_model_checkpoint):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(
            0,
            len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:
                                                       number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
            )
            shutil.rmtree(checkpoint)
