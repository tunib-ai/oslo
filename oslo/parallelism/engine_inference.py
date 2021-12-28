# UNDER DEVELOPMENT ...

import copyreg
import io
import pickle
from contextlib import suppress
from dataclasses import _is_dataclass_instance, asdict
from inspect import signature
from typing import Any

import torch
import torch.multiprocessing as mp
from transformers.file_utils import ModelOutput


class InferenceParallelEngine(object):
    __supported_fn__ = ["generate", "forward", "to", "cpu", "cuda"]

    """
    Inference Launcher of Parallelformers

    References:
        github: https://github.com/tunib-ai/parallelformers
        blog: https://tunib.tistory.com/entry/Parallelformers-Journey-to-deploying-big-modelsTUNiB
    """

    def __init__(self):
        self.processes = []
        self.parallel_mutexes = []
        self.inference_mutexes = []
        self.input_queues = []
        self.output_queues = []
        self.org_methods = {}

        with suppress(Exception):
            mp.set_start_method("spawn", force=True)
            # for shared memory

    def parallelize(self, model):
        model = model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False
            parameter.detace()

        for attr in self.__supported_fn__:
            if hasattr(model, attr):
                _attr = getattr(model, attr)
                self.org_methods[attr] = _attr

        try:
            parallel_mutex = mp.Event()
            inference_mutex = mp.Event()
            self.parallel_mutexes.append(parallel_mutex)
            self.inference_mutexes.append(inference_mutex)

        except Exception:
            pass

        for attr in self.__supported_fn__:
            if hasattr(model, attr):
                setattr(
                    model,
                    attr,
                    lambda *inputs, **kwargs: self.forward(
                        inputs=inputs,
                        kwargs=kwargs,
                        func=attr,
                    ),
                )

    def deparallelize(self):
        pass

    def forward(self, inputs, kwargs, func):
        pass


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


class InferenceProcess(mp.Process):
    __memory_fn__ = ["cuda", "cpu", "to"]

    def __init__(
        self,
        model_cls,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        parallel_mutex: mp.Event,
        inference_mutex: mp.Event,
        creation_fn,
    ):
        super().__init__()
        self.model_cls = model_cls
        self.model = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.parallel_mutex = parallel_mutex
        self.inference_mutex = inference_mutex
        self.creation_fn = creation_fn
        self.device = torch.cuda.current_device()

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

    def inference(self):
        while True:
            self.inference_mutex.wait()
            # hold gpu memory
            self.inference_mutex.clear()
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

    def initialize(self):
        self.model = self.creation_fn(self.model_cls)
        self.parallel_mutex.set()
        self.inference()
