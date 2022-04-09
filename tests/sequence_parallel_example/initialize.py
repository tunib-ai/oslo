import argparse
import os
import pprint
import torch
import torch.nn as nn
from colossalai.amp import AMP_TYPE, convert_to_amp
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.builder.builder import build_gradient_handler
from colossalai.context import Config, ConfigException, ParallelMode
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.core import global_context as gpc
from colossalai.engine import Engine
from colossalai.engine.ophooks import BaseOpHook
from colossalai.engine.schedule import NonPipelineSchedule, PipelineSchedule, InterleavedPipelineSchedule
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer.colossalai_optimizer import ColossalaiOptimizer
from colossalai.utils import (accumulate_gradient, get_current_device, is_using_ddp, is_using_pp, is_using_sequence,
                              sync_model_param)
from colossalai.utils.moe import sync_moe_model_param
from colossalai.zero import convert_to_zero_v2
from colossalai.zero.sharded_optim.sharded_optim_v2 import ShardedOptimizerV2
from pathlib import Path
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


def initialize(model: nn.Module,
               optimizer: Optimizer,
               criterion: Optional[_Loss] = None,
               train_dataloader: Optional[Iterable] = None,
               test_dataloader: Optional[Iterable] = None,
               lr_scheduler: Optional[_LRScheduler] = None,
               ophooks: Optional[List[BaseOpHook]] = None,
               verbose: bool = True) -> Tuple[Engine, DataLoader, DataLoader, _LRScheduler]:
    """Core function to wrap the essential training components with our functionality based on the config which is
    loaded into gpc.config.
    Args:
        model (:class:`torch.nn.Module` or Callbale): Your model instance or a function to build the model.
        optimizer (:class:`torch.optim.optimizer.Optimizer` or :class:`Type[torch.optim.optimizer]`):
            Your optimizer instance.
        criterion (:class:`torch.nn.modules.loss._Loss`, optional): Your criterion instance.
        train_dataloader (:class:`torch.utils.data.DataLoader`, optional): Dataloader for training.
        test_dataloader (:class:`torch.utils.data.DataLoader`, optional): Dataloader for testing.
        lr_scheduler (:class:`torch.nn.lr_scheduler._LRScheduler`, optional): Your lr scheduler instance, optional.
        verbose (bool, optional): Whether to print logs.
    Returns:
        Tuple (engine, train_dataloader, test_dataloader, lr_scheduler):
            A tuple of ``(engine, train_dataloader, test_dataloader, lr_scheduler)``
            where only ``engine`` could not be None.
    """
    # get logger
    logger = get_dist_logger()
    gpc.verbose = verbose

    # get config from gpc
    config = gpc.config

    # print config
    if verbose:
        logger.info(
            f"\n========== Your Config ========\n"
            f"{pprint.pformat(gpc.config)}\n"
            f"================================\n",
            ranks=[0])

    # cudnn
    cudnn_benchmark = config.get('cudnn_benchmark', True)
    cudnn_deterministic = config.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    if verbose:
        logger.info(f"cuDNN benchmark = {cudnn_benchmark}, deterministic = {cudnn_deterministic}", ranks=[0])

    # zero
    use_zero = hasattr(gpc.config, 'zero')
    if use_zero:
        zero_cfg = gpc.config.get('zero', None)
        if zero_cfg is not None:
            cfg_ = zero_cfg.copy()
        else:
            cfg_ = {}
        optimizer_config = zero_cfg.get('optimizer_config', None)
        model_config = zero_cfg.get('model_config', None)
        model, optimizer = convert_to_zero_v2(model,
                                              optimizer,
                                              model_config=model_config,
                                              optimizer_config=optimizer_config)

        logger.info("Initializing ZeRO model and optimizer finished!", ranks=[0])
        # FIXME() throw a warning if using zero with MP
        if gpc.get_world_size(ParallelMode.MODEL) > 1:
            logger.warning("ZeRO currently has not been tested with model parallelism.", ranks=[0])
    else:
        if isinstance(model, nn.Module):
            # first sync model across dp ranks
            print('current device: ', f'{get_current_device()}')
            model.to(get_current_device())
        elif isinstance(model, Callable):
            model = model().to(get_current_device())

        # added
        torch.distributed.barrier()

        # optimizer maybe a optimizer_cls
        logger.warning("Initializing an non ZeRO model with optimizer class")
        if isinstance(optimizer, Callable):
            optimizer = optimizer(model.parameters())

    # parameter sync
    if not use_zero:
        if is_using_sequence():
            # sync_model_param(model, ParallelMode.SEQUENCE_DP)
            parallel_mode = ParallelMode.SEQUENCE_DP
            if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
                for name, param in model.named_parameters():
                    ranks = gpc.get_ranks_in_group(parallel_mode)
                    torch.distributed.broadcast(param, src=ranks[0],
                                                # group=gpc.get_group(parallel_mode)
                                                )
        elif MOE_CONTEXT.is_initialized:
            sync_moe_model_param(model)
        elif is_using_ddp():
            sync_model_param(model, ParallelMode.DATA)
    else:
        logger.warning(
            "The parameters of models is not automatically synchronized.\n"
            "Please make sure that all parameters are the same in data parallel group.",
            ranks=[0])

    # check amp and zero
    fp16_cfg = gpc.config.get('fp16', None)

    if fp16_cfg is not None and fp16_cfg.mode is not None and use_zero:
        raise ConfigException(
            "It is not allowed to set fp16 and zero configuration in your config file at the same time")

    # clip grad norm
    clip_grad_norm = gpc.config.get('clip_grad_norm', 0.0)
    if clip_grad_norm > 0:
        if use_zero and zero_cfg is not None:
            raise ConfigException(
                "clip_grad_norm should be specified with zero, you should specify clip_grad in zero configuration")

    # initialize amp
    amp_mode = None
    if fp16_cfg is not None and fp16_cfg.mode is not None:
        cfg_ = fp16_cfg.copy()
        amp_mode = cfg_.pop('mode')
        if is_using_pp():
            assert amp_mode == AMP_TYPE.NAIVE, 'Pipeline only support NaiveAMP currently'
        if amp_mode == AMP_TYPE.NAIVE:
            cfg_['clip_grad_norm'] = clip_grad_norm
        model, optimizer, criterion = convert_to_amp(model=model,
                                                     optimizer=optimizer,
                                                     criterion=criterion,
                                                     mode=amp_mode,
                                                     amp_config=cfg_)

    # gradient handler
    gradient_handler_cfg = gpc.config.get('gradient_handler', None)
    if gradient_handler_cfg is None:
        # if gradient handler is not specified in the configuration file,
        # check in the following order
        # 1. if optimizer is ZERO, then use zero grad handler
        # 2. if dp size is larger than 1 and pipeline is not used, use pytorch ddp
        # 3. if using pipeline and dp size larger than 1, use data parallel grad handler
        if isinstance(optimizer, ShardedOptimizerV2):
            gradient_handler_cfg = [dict(type='ZeROGradientHandler')]
            if verbose:
                logger.info(
                    "Training with zero is detected, ZeROGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        elif is_using_ddp() and MOE_CONTEXT.is_initialized:
            gradient_handler_cfg = [dict(type='MoeGradientHandler')]
            if verbose:
                logger.info(
                    "Data parallel training is detected with moe parallel, MoeGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        elif is_using_sequence():
            model = DDP(model,
                        process_group=gpc.get_group(ParallelMode.SEQUENCE_DP),
                        device_ids=[torch.cuda.current_device()])
            if verbose:
                logger.info('Model is using torch.nn.parallel.DistributedDataParallel for Sequence Parallelism',
                            ranks=[0])
        elif is_using_ddp() and not is_using_pp() and amp_mode != AMP_TYPE.NAIVE:
            model = DDP(model, process_group=gpc.get_group(ParallelMode.DATA), device_ids=[torch.cuda.current_device()])
            if verbose:
                logger.info('Model is using torch.nn.parallel.DistributedDataParallel for Data Parallelism', ranks=[0])
        elif is_using_ddp():
            gradient_handler_cfg = [dict(type='DataParallelGradientHandler')]
            if verbose:
                logger.info(
                    "Data parallel training is detected when using pipeline parallel, "
                    "DataParallelGradientHandler is automatically "
                    "added even though not specified in the configuration",
                    ranks=[0])
        # add pipeline parallel gradient handler, if pipeline shared module is detected
        for param in model.parameters():
            if getattr(param, 'pipeline_shared_module_pg', None) is not None:
                if gradient_handler_cfg is None:
                    gradient_handler_cfg = [dict(type='PipelineSharedModuleGradientHandler')]
                else:
                    gradient_handler_cfg.append(dict(type='PipelineSharedModuleGradientHandler'))
                if verbose:
                    logger.info(
                        "pipeline_shared_module is detected, PipelineSharedModuleGradientHandler is automatically "
                        "added even though not specified in the configuration",
                        ranks=[0])
                break
    else:
        if not isinstance(gradient_handler_cfg, list):
            raise ConfigException(
                f"expected gradient_handler in the configuration file to be a list but got {type(gradient_handler_cfg)}"
            )

    # turn off sync buffer for NaiveAMPModel if using torch DDP and NaiveAMPModel at the same time
    # to avoid duplicated buffer synchronization
    if isinstance(model, DDP) and isinstance(model.module, NaiveAMPModel):
        model.module.sync_buffer = False

    # initialize schedule for engine
    if is_using_pp():
        tensor_shape = getattr(gpc.config, 'TENSOR_SHAPE', None)
        use_interleaved = hasattr(gpc.config, 'model') and hasattr(gpc.config.model, 'num_chunks')
        if use_interleaved:
            schedule = InterleavedPipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                                   gpc.config.model.num_chunks, tensor_shape=tensor_shape,
                                                   scatter_gather_tensors=True)
        else:
            schedule = PipelineSchedule(gpc.config.NUM_MICRO_BATCHES,
                                        tensor_shape=tensor_shape, scatter_gather_tensors=True)
    else:
        schedule = NonPipelineSchedule()

    if gradient_handler_cfg is None:
        gradient_handlers = None
        if verbose and not isinstance(model, DDP):
            logger.warning(
                "No PyTorch DDP or gradient handler is set up, please make sure you do not need "
                "to all-reduce the gradients after a training step.",
                ranks=[0])
    else:
        gradient_handlers = [build_gradient_handler(cfg, model, optimizer) for cfg in gradient_handler_cfg]

    # check if optimizer is ColossalaiOptimizer
    if not isinstance(optimizer, (ColossalaiOptimizer, ShardedOptimizerV2)):
        optimizer = ColossalaiOptimizer(optim=optimizer)

    # gradient accumulation
    grad_accum_size = gpc.config.get('gradient_accumulation', None)
    if grad_accum_size is not None:
        optimizer, train_dataloader, gradient_handlers, lr_scheduler = accumulate_gradient(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            accumulate_size=grad_accum_size,
            gradient_handlers=gradient_handlers,
            lr_scheduler=lr_scheduler)

    engine = Engine(model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    gradient_handlers=gradient_handlers,
                    clip_grad_norm=clip_grad_norm,
                    ophook_list=ophooks,
                    schedule=schedule)

    return engine, train_dataloader, test_dataloader, lr_scheduler
