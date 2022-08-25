# Copyright 2021 TUNiB Inc.
import copyreg
import io
import os
import pickle
import random
import traceback
from dataclasses import _is_dataclass_instance, asdict
from inspect import signature
from time import time
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dacite import Config, from_dict
from transformers.file_utils import ModelOutput


class DeploymentParallelEngine(object):
    """
    Deployment Launcher of Parallelformers

    Args:
        seed (int): random seed value

    References:
        github: https://github.com/tunib-ai/parallelformers
        blog: https://tunib.tistory.com/entry/Parallelformers-Journey-to-deploying-big-modelsTUNiB
    """

    def __init__(self, model, **kwargs):
        self.model = model.eval()
        self.kwargs = kwargs

        self.processes = []
        self.parallelization_mutexes = []
        self.reqeust_mutexes = []
        self.input_queues = []
        self.output_queues = []
        self.orig_methods = {}

        os.environ["WORLD_SIZE"] = str(
            kwargs.get("tensor_parallel_size", 1)
            * kwargs.get("pipeline_parallel_size", 1)
        )
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"

    def replace_methods(self, method: str) -> None:
        """
        Intercept the flow by changing some methods (e.g. forward, generate, ...)
        in the model to `self.forward` methods.

        Args:
            method (str): name of method
        """

        setattr(
            self.model,
            method,
            lambda *inputs, **kwargs: self.forward(
                inputs=inputs,
                kwargs=kwargs,
                func=method,
            ),
        )

    def parallelize(self):
        for attr in DeploymentProcess.__supported_fn__:
            if hasattr(self.model, attr):
                fn = getattr(self.model, attr)
                self.orig_methods[attr] = fn

        try:
            for rank in range(int(os.environ["WORLD_SIZE"])):
                parallelization_mutex = mp.Event()
                reqeust_mutex = mp.Event()
                self.parallelization_mutexes.append(parallelization_mutex)
                self.reqeust_mutexes.append(reqeust_mutex)

                input_queue = mp.Queue()
                output_queue = mp.Queue()
                self.input_queues.append(input_queue)
                self.output_queues.append(output_queue)

                process = DeploymentProcess(
                    rank=rank,
                    model=self.model,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    parallelization_mutex=parallelization_mutex,
                    request_mutex=reqeust_mutex,
                    **self.kwargs,
                )

                process.daemon = True
                process.start()
                self.processes.append(process)

            for mutex in self.parallelization_mutexes:
                mutex.wait()

            for attr in DeploymentProcess.__supported_fn__:
                if hasattr(self.model, attr):
                    self.replace_methods(attr)

        except BaseException:
            traceback.print_exc()
            self._deparallelize()

    def _deparallelize(self) -> None:
        """
        Remove all methods registered in the model
        and join all GPU processes to main process.
        """

        if hasattr(self, "orig_methods"):
            for k, v in self.orig_methods.items():
                setattr(self.model, k, v)

        if hasattr(self, "processes"):
            for process in self.processes:
                process.join()

    @staticmethod
    def _deallocate(item):
        if torch.is_tensor(item) and item.is_cuda:
            item.cpu()

        elif isinstance(item, list) or isinstance(item, tuple):
            for i in item:
                if torch.is_tensor(i) and i.is_cuda:
                    i.cpu()

        elif isinstance(item, dict):
            for i in item:
                if torch.is_tensor(item[i]) and item[i].is_cuda:
                    item[i].cpu()

        return item

    def forward(
        self,
        inputs: Any,
        kwargs: Dict,
        func: str,
    ):
        try:
            for i_mutex, i_queue in zip(
                self.reqeust_mutexes,
                self.input_queues,
            ):
                inputs = self._deallocate(inputs)

                for k in kwargs:
                    kwargs[k] = self._deallocate(kwargs[k])

                i_queue.put((inputs, kwargs, func))
                i_mutex.set()
                # producer part

            if func in ["to", "cpu", "cuda"]:
                self._deparallelize()

                if func == "cpu":
                    self.model = self.model.cpu(*inputs, **kwargs)
                elif func == "cuda":
                    self.model = self.model.cuda(*inputs, **kwargs)
                else:
                    self.model = self.model.to(*inputs, **kwargs)

                return self.model
            else:
                outputs = []
                for o_queue in self.output_queues:
                    output = o_queue.get()
                    outputs.append(output)

                # use output of rank:0
                final_output = outputs[0]

                # non-picklable object to original dataclass
                if (
                    isinstance(final_output, dict)
                    and "orig_dataclass_type" in final_output
                ):
                    orig_dataclass_type = final_output["orig_dataclass_type"]
                    del final_output["orig_dataclass_type"]

                    final_output = from_dict(
                        orig_dataclass_type,
                        final_output,
                        config=Config(check_types=False),
                    )

                return final_output

        except BaseException:
            traceback.print_exc()
            self._deparallelize()


class ForkingPickler(pickle.Pickler):
    """Copy of ForkingPickler of `multiprocessing` module"""

    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __init__(self, *args):
        """Constructor of ForkingPickler"""
        super().__init__(*args)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce) -> None:
        """Register reduce methods for multiprocessing"""
        cls._extra_reducers[type] = reduce

    @classmethod
    def dumps(cls, obj: Any, protocol=None) -> memoryview:
        """Dump objects for multiprocessing"""
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        return buf.getbuffer()

    loads = pickle.loads


class DeploymentProcess(mp.Process):
    __memory_fn__ = ["cuda", "cpu", "to"]
    __supported_fn__ = ["generate", "forward"] + __memory_fn__

    def __init__(
        self,
        rank: int,
        model: nn.Module,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        parallelization_mutex: mp.Event,
        request_mutex: mp.Event,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.orig_methods = {}
        self.device = None

        self.model = model.eval()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.parallelization_mutex = parallelization_mutex
        self.request_mutex = request_mutex
        self.kwargs = kwargs
        self.from_split_checkpoint_files = kwargs.get(
            "from_split_checkpoint_files",
            False,
        )

    def check_picklable(self, obj: Any) -> Any:
        """
        Check object is picklable.
        If it is not picklable, this method will change the dataclass instance to a dictionary.
        It is is not dataclass raise exception.

        Args:
            obj (Any): object to check picklable

        Returns:
            Any: picklable object
        """
        try:
            pickle.loads(ForkingPickler.dumps(obj).tobytes())
        except BaseException:
            if hasattr(self.model, "mpu") and self.model.is_pipeline_parallelized():
                obj = [_ for _ in obj][0]

            if _is_dataclass_instance(obj) or isinstance(obj, ModelOutput):
                _obj = asdict(obj)
                _obj["orig_dataclass_type"] = obj.__class__
                obj = _obj

            else:
                raise Exception(
                    f"Type '{obj.__class__}' can't be pickled. "
                    f"Please check type of model output !"
                )

        return obj

    @torch.no_grad()
    def wait_request(self):
        if not self.kwargs.get("seed", None):
            seed = torch.tensor(int(time())).cuda()
            dist.broadcast(seed, src=0)
            seed = seed.item()
        else:
            seed = self.kwargs.get("seed")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        while True:
            self.request_mutex.wait()
            # hold gpu memory
            self.request_mutex.clear()
            # clear mutex to forward model

            _inputs, _kwargs = [], {}
            inputs, kwargs, fn_code = self.input_queue.get()
            # get inputs from input queue

            for i in inputs:
                if torch.is_tensor(i):
                    i = i.to(self.device)
                _inputs.append(i)

            for k in kwargs:
                if torch.is_tensor(kwargs[k]):
                    kwargs[k] = kwargs[k].to(self.device)
                _kwargs[k] = kwargs[k]

            if hasattr(self.model, "mpu") and self.model.is_pipeline_parallelized():
                from oslo.parallelism.engine_pipeline import (
                    PipelineParallelEngine,
                )

                # turn off pipelining
                batch_size = PipelineParallelEngine.guess_batch_size(kwargs)
                self.model.set_micro_batch_size(batch_size)

            function_ = getattr(self.model, fn_code)
            n_params = len(signature(function_).parameters)

            if n_params > 0:
                outputs = function_(*_inputs, **_kwargs)
            else:
                outputs = function_()

            if fn_code in self.__memory_fn__:
                # free all the gpu memory
                break

            outputs = self.check_picklable(outputs)
            self.output_queue.put(outputs)

    def run(self) -> None:
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
        os.environ["LOCAL_RANK"] = str(self.rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(
            self.kwargs.get("tensor_parallel_size")
            * self.kwargs.get("pipeline_parallel_size")
        )
        torch.cuda.set_device(self.rank)
        self.device = torch.cuda.current_device()

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.model._exec_both_engines(model=self.model, **self.kwargs)

        if self.from_split_checkpoint_files is True:
            self.model._load_split_checkpoint_files(model=self.model, **self.kwargs)

        self.parallelization_mutex.set()
        self.wait_request()

        for parameter in self.model.parameters():
            parameter.requires_grad = False
            parameter.detach()
