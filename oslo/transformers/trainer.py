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

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerBase

# from .modeling_utils import PreTrainedModel, unwrap_model
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .training_args import ParallelMode, TrainingArguments
from .trainer_utils import (
    set_seed,
    has_length,
    EvalPrediction,
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
    LabelSmoother,)
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

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

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

        if args is None:
            # No Arguments passed
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        set_seed(self.args.seed)
        self.hp_name = None  # TODO ?
        self.is_in_train = False  # TODO ?

        # TODO check - should we support memory tracker?
        # # memory metrics - must set up as early as possible
        # self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        # self._memory_tracker.start()

        self.is_in_train = False

        # set log level
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        # # Removed
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
        """
        self.sharded_ddp = None
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

        # TODO - Task F: Create all CALLBACKS functions (includes TrainerCallback)
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(callbacks, self.model,
                                                self.tokenizer, self.optimizer,
                                                self.lr_scheduler)
        # self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.callback_handler.add_callback(DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False  # TODO check 어디에 쓰이는거지...?

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
        self.use_apex = False
        self.use_amp = False
        self.do_grad_scaling = False

        # TODO - Task G: all about mixed precision
        if args.fp16 or args.bf16:
            if args.half_precision_backend == "auto":
                if _is_native_amp_available:
                    args.half_precision_backend = "amp"
                else:
                    if args.bf16:
                        raise ValueError(
                            "Tried to use `bf16` but native amp is not available"
                        )
                    else:
                        args.half_precision_backend = "apex"
            logger.info(
                f"Using {args.half_precision_backend} half precision backend")

            self.use_amp = True
            self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
            self.do_grad_scaling = True
            # TODO
            # if self.sharded_ddp is not None:
            #     self.scaler = ShardedGradScaler()
            #
            # else:
            #     self.scaler = torch.cuda.amp.GradScaler()

        # TODO - Task H: LabelSmoother
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        # TODO what is ...?
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        self.control = TrainerControl()

        # TODO check should we support count flos ?
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None  # TODO ?
        self.use_tune_checkpoints = False  # TODO ?

        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(
            self.args, self.state, self.control)

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

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
