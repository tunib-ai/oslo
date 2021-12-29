# Copyright 2021 TUNiB Inc.
import inspect
from collections.abc import Iterable
from functools import partial
from typing import List

import torch
import torch.distributed as dist
from torch import Size, Tensor, nn

from oslo.parallelism.mpu import MPU, Layer, LayerPolicy

NoneType = type(None)


class PipelineParallelEngine(object):
    """
    Pipeline model parallelism engine designed by TUNiB.

    We use the 1F1B scheduling described in Pipedream Flush.
    https://www.deepspeed.ai/assets/images/pipe-schedule.png

    Notes:
        There are three new concepts in our engine that make our engine scalable.

        1. Micro Loss Tensor:
        With the existing frameworks such as DeepSpeed and Megatron-LM,
        user can't manipluate loss of micro batches because they hide all the pipeline scheduling.
        To solve this problem, we propose the concept of micro loss tensor, and this tensor can be manipulated by users.
        This gives you the freedom to use frameworks like DeepSpeed, AMP, etc.

        2. Universal P2P Communicator:
        The P2P communicator in the exsting frameworks such as DeepSpeed and Megatron-LM can send or receive only tensor type object.
        But the ``ModelOutput`` object in the Hugging Face Transformers has values of various types. That's why we need to make p2p communicator more scalable.
        So, we made new p2p communicator that supports almost all types in python such as ``int``, ``str``, ``bool``, ``tuple``,
        and the nested data types such as ``Tuple[Dict[str, Tensor], List[bool, bool]]`` can be sended or received.

        3. Function splitting rather than model splitting:
        The existing frameworks such as DeepSpeed and TorchGPipe need ``nn.Sequential`` module to use pipeline parallelism.
        So, user have needed to make model to ``nn.Sequential`` type.
        But this process is cumbersome. We split forward function into 6 sub-functions rather than splitting model.
        By function splitting, user don't need to care about model architecture anymore.

    Args:
        mpu (MPU): model parallel unit
        policy (LayerPolicy): layer policy
        micro_batch_size (int): micro batch size

    Examples:
        # Example 1:
        Pipeline parallelism without other framework
        >>> for output in model(...):
        >>>     output.loss.backward()
        >>> optimizer.step()

        # Example 2:
        Pipeline parallelism with Microsoft DeepSpeed
        >>> for output in engine(...):
        >>>     engine.backward(output.loss)
        >>> engine.step(loss)

        # Example 3:
        Pipeline parallelism with NVIDIA AMP
        >>> for output in model(...):
        >>>     with amp.scale_loss(output.loss, optimizer) as scaled_loss:
        >>>         scaled_loss.backward()
        >>> optimizer.step(loss)
    """

    def __init__(
        self,
        mpu: MPU,
        policy: LayerPolicy,
        micro_batch_size: int,
        head_layers: List[Layer] = None,
    ):
        # model parallel utils
        self.mpu = mpu
        self.policy = policy
        self.engine = None
        self.p2p = P2PCommunicator(self.mpu)
        self.device = torch.cuda.current_device()
        self.head_layers = head_layers if head_layers is not None else []
        self.zero_stage = None

        # functions utils
        self.loss_fn = None
        self.organize_fn = None
        self.forward_fns = []

        # batch size utils
        self.batch_size = None
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = None
        self.micro_batches = None
        self.micro_offset = 0

        # stage utils
        self.num_stages = mpu.get_pipeline_parallel_world_size()
        self.stage_id = mpu.get_pipeline_parallel_rank()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1
        self.local_start = None
        self.local_stop = None

        # buffer utils
        self.num_buffers = 0
        self.buffers = {
            "inputs": [],
            "outputs": [],
            "grads": [],
            "embeddings": {},
            "modules": nn.ModuleList(),
        }

    def set_micro_batch_size(self, micro_batch_size):
        """For generation_utils.py"""
        self.micro_batch_size = micro_batch_size

    def _parallelize_first_stage(self, model: nn.Module):
        """
        Parallelize pre-block layers and word embedding layers

        Args:
            model (nn.Module): model object
        """

        # word embedding layer
        for embedding in self.policy.word_embedding(model, model.config):
            if embedding.module is not None:
                self.buffers["modules"].append(embedding.module)
            if embedding.weight is not None:
                embedding.weight.data = embedding.weight.to(self.device)
            if embedding.bias is not None:
                embedding.bias.data = embedding.bias.to(self.device)

        # pre-block layers (e.g. positional emb)
        for preblock_layer in self.policy.preblock_layers(model, model.config):
            if preblock_layer.module is not None:
                self.buffers["modules"].append(preblock_layer.module)
            if preblock_layer.weight is not None:
                preblock_layer.weight.data = preblock_layer.weight.to(self.device)
            if preblock_layer.bias is not None:
                preblock_layer.bias.data = preblock_layer.bias.to(self.device)

    def _parallelize_last_stage(self, model: nn.Module):
        """
        Parallelize post-block layers and head layers

        Args:
            model (nn.Module): model object
        """

        # post-block layers (e.g. last layernorm)
        for postblock_layer in self.policy.postblock_layers(model, model.config):
            if postblock_layer.module is not None:
                self.buffers["modules"].append(postblock_layer.module)
            if postblock_layer.weight is not None:
                postblock_layer.weight.data = postblock_layer.weight.to(self.device)
            if postblock_layer.bias is not None:
                postblock_layer.bias.data = postblock_layer.bias.to(self.device)

        # additional head layers (e.g. lm head)
        for head_layers in self.head_layers:
            if head_layers.module is not None:
                self.buffers["modules"].append(head_layers.module)
            if head_layers.weight is not None:
                head_layers.weight.data = head_layers.weight.to(self.device)
            if head_layers.bias is not None:
                head_layers.bias.data = head_layers.bias.to(self.device)

    def _collect_tied_layer(self):
        """Collect tied embedding layers to do all-reduce gradient"""
        for head_layers in self.head_layers:
            if head_layers.tied_embedding is not None:
                if head_layers.weight is not None:
                    self.buffers["embeddings"][
                        head_layers.weight
                    ] = head_layers.tied_embedding.weight
                if head_layers.bias is not None:
                    self.buffers["embeddings"][
                        head_layers.bias
                    ] = head_layers.tied_embedding.bias

    def parallelize(self, model: nn.Module):
        """
        Inter-layer model parallelize with pipelining

        Args:
            model (nn.Module): model object
        """
        self.organize_fn = model.organize_fn
        self.loss_fn = model.loss_fn

        base_model = model.base_model
        block_layers = self.policy.block_layers(
            base_model,
            base_model.config,
        )
        num_layers = len(block_layers)

        # compute range of partition for current rank
        partitions = self.mpu.make_pipeline_partition(
            num_items=num_layers,
            num_parts=self.num_stages,
        )

        self.local_start = partitions[self.stage_id]
        self.local_stop = partitions[self.stage_id + 1]

        # parallelize pre-block and word embedding layers
        if self.mpu.is_pipeline_first_stage():
            self._parallelize_first_stage(base_model)
            self.forward_fns.append(model.preblock_fn)
            self._collect_tied_layer()

        # parallelize block layers
        for i, layer in enumerate(block_layers):
            if self.local_start <= i < self.local_stop:
                self.forward_fns.append(
                    partial(
                        model.block_fn,
                        layer_id=i,
                    )
                )

                # move all layers to gpus.
                parameters = (
                    self.policy.attn_qkv(layer, model.config)
                    + self.policy.attn_out(layer, model.config)
                    + self.policy.attn_norm(layer, model.config)
                    + self.policy.mlp_in(layer, model.config)
                    + self.policy.mlp_out(layer, model.config)
                    + self.policy.mlp_norm(layer, model.config)
                    + self.policy.copy_to_all(layer, model.config)
                )

                for param in parameters:
                    if param.module is not None:
                        self.buffers["modules"].append(param.module)
                    if param.weight is not None:
                        param.weight.data = param.weight.to(self.device)
                    if param.bias is not None:
                        param.bias.data = param.bias.to(self.device)

        # parallelize post-block and head layers
        if self.mpu.is_pipeline_last_stage():
            self._parallelize_last_stage(base_model)
            self.forward_fns.append(model.postblock_fn)
            self.forward_fns.append(model.head_fn)
            self._collect_tied_layer()

        setattr(model, "orig_forward", model.forward)
        setattr(model, "forward", self.forward)
        setattr(model, "set_micro_batch_size", self.set_micro_batch_size)

        setattr(self.buffers["modules"], "forward", self.forward)
        setattr(model.__class__, "pipe_modules", self.buffers["modules"])
        # Hack: If we register pipe modules to ``self``, it will be a member of state dict.
        # So we register pipe modules to ``__class__`` to avoid checkpoint saving.

    def forward(self, *args, **kwargs):
        """Do forward pass with pipeline scheduling"""

        assert len(args) == 0, (
            "Pipeline parallel model only supports ``**kwargs`` input (keyword arguments). "
            "If you write code like ``model(input_ids, labels)``, "
            "please modify your code like ``model(input_ids=input_ids, labels=labels)``."
        )

        if self.zero_stage is None:
            self._guess_zero_stage()

        if self.zero_stage > 1:
            raise RuntimeError(
                "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism. "
                "See https://github.com/microsoft/DeepSpeed/issues/1110 for more details."
            )

        self.micro_offset = 0
        self.micro_batches = self._split_batches(kwargs)
        self._reserve_buffers(self.num_micro_batches)

        for i in range(self.num_micro_batches):
            if self.mpu.is_pipeline_first_stage():
                self._exec_load_micro_batch(i)

            if not self.mpu.is_pipeline_first_stage():
                self._exec_recv_activations(i)

            self._exec_forward_pass(i)

            if not self.mpu.is_pipeline_last_stage():
                self._exec_send_activations(i)

            yield self._exec_postprocess(i)

        self._exec_allreduce_embedding()

        for i in range(self.num_micro_batches):
            self._free_buffers("inputs", i)
            self._free_buffers("outputs", i)
            self._free_buffers("grads", i)

    @staticmethod
    def guess_batch_size(kwargs):
        """Guess global batch size dynamically from user input"""
        for key in ["input_ids", "attention_mask", "labels"]:
            if kwargs.get(key, None) is not None:
                assert torch.is_tensor(
                    kwargs.get(key)
                ), f"Param ``{key}`` must be ``torch.Tensor`` that has shape like (batch_size, ...)."

                return kwargs.get(key).size(0)

        kwargs_types = {k: type(v).__qualname__ for k, v in kwargs.items()}

        raise ValueError(
            f"You must at least input one of ``input_ids``, ``attention_mask`` or ``labels``. "
            f"But you didn't input any of them. Please double check your input: {kwargs_types}."
        )

    def _split_batches(self, batches):
        """Split mini-batches to micro-batches"""
        self.batch_size = self.guess_batch_size(batches)
        assert self.batch_size % self.micro_batch_size == 0, (
            "``micro_batch_size`` must be divisible by batch size. "
            f"currently, ``micro_batch_size`` is {self.micro_batch_size}. "
            f"but batch size is {self.batch_size}."
        )

        self.num_micro_batches = self.batch_size // self.micro_batch_size
        _micro_batches = [{} for _ in range(self.num_micro_batches)]

        for k, v in batches.items():
            if torch.is_tensor(v):
                if v.size(0) == self.batch_size:
                    micro_batch = v.chunk(self.num_micro_batches, dim=0)
                    for i, m in enumerate(micro_batch):
                        _micro_batches[i][k] = m
                else:
                    for i in range(self.num_micro_batches):
                        _micro_batches[i][k] = v
            else:
                for i in range(self.num_micro_batches):
                    _micro_batches[i][k] = v

        return _micro_batches

    @staticmethod
    def _zero_grads(inputs):
        """Make all the gradient in tensors to zero"""

        def _zero_grad(_input):
            if torch.is_tensor(_input):
                if _input.is_leaf and _input.grad is not None:
                    _input.grad.data.zero_()
            elif isinstance(_input, dict):
                for v in _input.values():
                    _zero_grad(v)
            elif isinstance(_input, Iterable):
                for item in _input:
                    _zero_grad(item)

        _zero_grad(inputs)

        return inputs

    def _reserve_buffers(self, num_buffers):
        """Allocate pipeline buffers when we start training"""
        if self.num_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_buffers
        reserved_key = ["inputs", "outputs", "grads"]

        for key in self.buffers:
            if key in reserved_key:
                self.buffers[key] = [{} for _ in range(num_added)]

        self.num_buffers = num_buffers

    def _exec_load_micro_batch(self, buffer_id):
        """Load micro-batch from split mini-batches"""
        micro_batch = self.micro_batches[self.micro_offset]
        if self.mpu.is_pipeline_first_stage():
            loaded = {}
            for k, v in micro_batch.items():
                if torch.is_tensor(v):
                    v = v.clone().detach().to(self.device)
                    v.requires_grad = v.is_floating_point()
                loaded[k] = v

            self.buffers["inputs"][buffer_id] = loaded

    def _exec_forward_pass(self, buffer_id):
        """Do forward pass"""
        self.micro_offset += 1
        inputs = self.buffers["inputs"][buffer_id]
        inputs = self._zero_grads(inputs)

        for func in self.forward_fns:
            inputs = func(**inputs)

        outputs = inputs
        # final output of forward fns

        self.buffers["outputs"][buffer_id] = outputs
        # send to next stage

    def _exec_send_activations(self, buffer_id):
        """Send activation tensors to next stage"""
        self.p2p.send_dict(self.buffers["outputs"][buffer_id], self.next_stage)

    def _exec_recv_activations(self, buffer_id):
        """Receive activation tensors from previous stage"""
        self.buffers["inputs"][buffer_id] = self.p2p.recv_dict(self.prev_stage)

    def _exec_postprocess(self, buffer_id):
        """Do postprocess after forward scheduling"""
        outputs = self.buffers["outputs"][buffer_id]

        if self.mpu.is_pipeline_last_stage():
            outputs["loss"] = self.loss_fn(**outputs)
            final_outputs = outputs
        else:
            final_outputs = {}

        final_outputs = self.p2p.ring_exchange_backward(final_outputs)
        # do ring exchange to make every stage have same output.
        # we need to make user don't care about stage or rank ;)

        if final_outputs["loss"] is not None:
            final_outputs["loss"].__class__ = MicroLossTensor
            final_outputs["loss"].set_arguments(self.p2p, self.buffers, buffer_id)

        return self.organize_fn(**final_outputs)

    def _exec_allreduce_embedding(self):
        """Perform all-reduce for tied embedding layers"""
        for head, embedding in self.buffers["embeddings"].items():
            if head.grad is not None and embedding.grad is not None:
                # all-reduce gradient between first stage and last stage.
                dist.all_reduce(
                    tensor=embedding.grad,
                    group=self.mpu.get_embedding_tied_group(),
                )
                dist.all_reduce(
                    tensor=head.grad,
                    group=self.mpu.get_embedding_tied_group(),
                )

    def _free_buffers(self, buffer_key, buffer_id):
        """Free pipeline buffers to save memory usage"""
        self.buffers[buffer_key][buffer_id] = {}

    def _guess_zero_stage(self):
        """
        ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism
        So we prohibit usage of ZeRO-2 or higher when user uses pipeline parallelism

        See Also:
            https://github.com/microsoft/DeepSpeed/issues/1110
        """
        try:
            from deepspeed.runtime.engine import DeepSpeedEngine

            call_stacks = [item[0].f_locals for item in inspect.stack()]

            for call_stack in call_stacks:
                for obj in call_stack.values():
                    if isinstance(obj, DeepSpeedEngine):
                        if obj.zero_optimization():
                            self.zero_stage = obj.zero_optimization_stage()
                            return

            self.zero_stage = 0

        except ImportError:
            self.zero_stage = 0


class PipelineDeparallelEngine(object):
    """
    Pipeline model deparallelism engine designed by TUNiB.

    Args:
        mpu (MPU): model parallel unit
        policy (LayerPolicy): layer policy
        head_layers (List[Layer]): head layers
    """

    def __init__(self, mpu, policy, head_layers):
        self.mpu = mpu
        self.policy = policy
        self.num_stages = mpu.get_pipeline_parallel_world_size()
        self.stage_id = mpu.get_pipeline_parallel_rank()
        self.first_stage = 0
        self.last_stage = self.num_stages - 1
        self.head_layers = head_layers if head_layers is not None else []
        self.stage_to_rank = mpu.get_pipeline_model_parallel_global_ranks_per_group()

    def _broadcast_param(self, layer, src):
        """Collect parameters from all the ranks"""
        if layer.weight is not None:
            weight = layer.weight.data.clone().cuda()
            dist.broadcast(
                weight,
                src=self.stage_to_rank[src],
                group=self.mpu.get_pipeline_parallel_group(),
            )
            layer.weight.data = weight

        if layer.bias is not None:
            bias = layer.bias.data.clone().cuda()
            dist.broadcast(
                bias,
                src=self.stage_to_rank[src],
                group=self.mpu.get_pipeline_parallel_group(),
            )
            layer.bias.data = bias

    def _deparallelize_first_stage(self, model: nn.Module):
        """
        Deparallelize pre-block layers and word embedding layers

        Args:
            model (nn.Module): model object
        """

        for embedding in self.policy.word_embedding(model, model.config):
            self._broadcast_param(embedding, src=self.first_stage)
        for preblock_layer in self.policy.preblock_layers(model, model.config):
            self._broadcast_param(preblock_layer, src=self.first_stage)

    def _deparallelize_last_stage(self, model: nn.Module):
        """
        Deparallelize post-block layers and head layers

        Args:
            model (nn.Module): model object
        """

        for postblock_layer in self.policy.postblock_layers(model, model.config):
            self._broadcast_param(postblock_layer, src=self.last_stage)
        for head_layers in self.head_layers:
            self._broadcast_param(head_layers, src=self.last_stage)

    def deparallelize(self, model: nn.Module):
        """
        Inter-layer model deparallelize

        Args:
            model (nn.Module): model object
        """
        base_model = model.base_model
        block_layers = self.policy.block_layers(
            base_model,
            base_model.config,
        )
        num_layers = len(block_layers)

        # compute range of partition for current rank
        partitions = self.mpu.extend_pipeline_partition(
            self.mpu.make_pipeline_partition(
                num_items=num_layers,
                num_parts=self.num_stages,
            )
        )

        # deparallelize pre-block and word embedding layers
        self._deparallelize_first_stage(base_model)

        # deparallelize pipe layers
        for layer, stage in zip(block_layers, partitions):
            parameters = (
                self.policy.attn_qkv(layer, model.config)
                + self.policy.attn_out(layer, model.config)
                + self.policy.attn_norm(layer, model.config)
                + self.policy.mlp_in(layer, model.config)
                + self.policy.mlp_out(layer, model.config)
                + self.policy.mlp_norm(layer, model.config)
                + self.policy.copy_to_all(layer, model.config)
            )

            for param in parameters:
                self._broadcast_param(param, src=stage)

        # deparallelize post-block and head layers
        self._deparallelize_last_stage(base_model)

        # forward function restoring
        if hasattr(model, "orig_forward"):
            setattr(model, "forward", getattr(model, "orig_forward", model.forward))


class MicroLossTensor(torch.Tensor):
    """
    Micro loss tensor that supports backward scheduling

    We use the 1F1B scheduling described in Pipedream Flush.
    https://www.deepspeed.ai/assets/images/pipe-schedule.png

    Notes:
        There are three new concepts in our engine that make our engine scalable.

        1. Micro Loss Tensor:
        With the existing frameworks such as DeepSpeed and Megatron-LM,
        user can't manipluate loss of micro batches because they hide all the pipeline scheduling.
        To solve this problem, we propose the concept of micro loss tensor, and this tensor can be manipulated by users.
        This gives you the freedom to use frameworks like DeepSpeed, AMP, etc.

        2. Universal P2P Communicator:
        The P2P communicator in the exsting frameworks such as DeepSpeed and Megatron-LM can send or receive only tensor type object.
        But the ``ModelOutput`` object in the Hugging Face Transformers has values of various types. That's why we need to make p2p communicator more scalable.
        So, we made new p2p communicator that supports almost all types in python such as ``int``, ``str``, ``bool``, ``tuple``,
        and the nested data types such as ``Tuple[Dict[str, Tensor], List[bool, bool]]`` can be sended or received.

        3. Function splitting rather than model splitting:
        The existing frameworks such as DeepSpeed and TorchGPipe need ``nn.Sequential`` module to use pipeline parallelism.
        So, user have needed to make model to ``nn.Sequential`` type.
        But this process is cumbersome. We split forward function into 6 sub-functions rather than splitting model.
        By function splitting, user don't need to care about model architecture anymore.

    Examples:
        # Example 1:
        Pipeline parallelism without other framework
        >>> for output in model(...):
        >>>     output.loss.backward()
        >>> optimizer.step()

        # Example 2:
        Pipeline parallelism with Microsoft DeepSpeed
        >>> for output in engine(...):
        >>>     engine.backward(output.loss)
        >>> engine.step(loss)

        # Example 3:
        Pipeline parallelism with NVIDIA AMP
        >>> for output in model(...):
        >>>     with amp.scale_loss(output.loss, optimizer) as scaled_loss:
        >>>         scaled_loss.backward()
        >>> optimizer.step(loss)
    """

    p2p = None
    mpu = None
    buffers = None
    buffer_id = None
    stage_id = None
    num_stages = None
    prev_stage = None
    next_stage = None

    @classmethod
    def set_arguments(cls, p2p, buffers, buffer_id):
        """Set arguments to schedule backward pass"""
        cls.p2p = p2p
        cls.mpu = p2p.mpu
        cls.buffers = buffers
        cls.buffer_id = buffer_id
        cls.stage_id = p2p.mpu.get_pipeline_parallel_rank()
        cls.num_stages = p2p.mpu.get_pipeline_parallel_world_size()
        cls.prev_stage = cls.stage_id - 1
        cls.next_stage = cls.stage_id + 1

    def backward(self, **kwargs):
        """
        Do backward pass with scheduling
        This is end-point method that user uses.

        Examples:
            >>> loss = output.loss
            >>> loss.backward()
        """

        if not self.mpu.is_pipeline_last_stage():
            self._exec_recv_gradient(self.buffer_id)

        self._exec_backward_pass(self.buffer_id)

        if not self.mpu.is_pipeline_first_stage():
            self._exec_send_gradient(self.buffer_id)

    def _exec_send_gradient(self, buffer_id):
        """Send gradient tensors to previous stage"""
        assert len(self.buffers["inputs"][buffer_id]) > 0, (
            "Input buffers of pipeline parallelized model are empty. "
            "Did you call ``loss.backward()`` outside of micro batch for loop context? "
            "You must call ``loss.backward()`` inside of micro batch for loop context. "
            "Please check your code."
        )

        for key, val in self.buffers["inputs"][buffer_id].items():
            if torch.is_tensor(val) and val.grad is not None:
                self.buffers["grads"][buffer_id][key] = val.grad

        gradient = self.buffers["grads"][buffer_id]
        self.p2p.send_dict(gradient, self.prev_stage)

        # free all the buffers to reduce memory usage
        self._free_buffers("inputs", buffer_id)
        self._free_buffers("outputs", buffer_id)
        self._free_buffers("grads", buffer_id)

    def _exec_recv_gradient(self, buffer_id):
        """Receive gradient tensors from next stage"""
        self.buffers["grads"][buffer_id] = self.p2p.recv_dict(self.next_stage)

    def _exec_backward_pass(self, buffer_id, **kwargs):
        if self.mpu.is_pipeline_last_stage():
            super().backward(**kwargs)
            return

        assert len(self.buffers["outputs"][buffer_id]) > 0, (
            "Input buffers of pipeline parallelized model are empty. "
            "Did you call ``loss.backward()`` outside of micro batch for loop context? "
            "You must call ``loss.backward()`` inside of micro batch for loop context. "
            "Please check your code."
        )

        outputs = self.buffers["outputs"][buffer_id]
        grads = self.buffers["grads"][buffer_id]
        trainable = [outputs[k] for k in outputs if k in grads]

        assert len(trainable) == len(grads), (
            "the number of received gradient tensor and trainable tensors are different. "
            "please check your model output"
        )

        torch.autograd.backward(
            tensors=tuple(trainable),
            grad_tensors=tuple(grads.values()),
        )

    def _free_buffers(self, buffer_key, buffer_id):
        """Free pipeline buffers to save memory usage"""
        self.buffers[buffer_key][buffer_id] = {}


class P2PCommunicator(object):
    """
    P2P communicator that can communicate various data types.

    Args:
        mpu (MPU): model parallel unit
    """

    def __init__(self, mpu: MPU):
        self.mpu = mpu
        self.device = torch.device(torch.cuda.current_device())
        self.num_stages = mpu.get_pipeline_parallel_world_size()
        self.stage_id = mpu.get_pipeline_parallel_rank()
        self.INSTRUCTIONS = {
            bool: {"send": self.send_bool, "recv": self.recv_bool},
            int: {"send": self.send_int, "recv": self.recv_int},
            float: {"send": self.send_float, "recv": self.recv_float},
            complex: {"send": self.send_complex, "recv": self.recv_complex},
            str: {"send": self.send_str, "recv": self.recv_str},
            type: {"send": self.send_type, "recv": self.recv_type},
            list: {"send": self.send_list, "recv": self.recv_list},
            tuple: {"send": self.send_tuple, "recv": self.recv_tuple},
            set: {"send": self.send_set, "recv": self.recv_set},
            dict: {"send": self.send_dict, "recv": self.recv_dict},
            NoneType: {"send": self.send_none, "recv": self.recv_none},
            Size: {"send": self.send_size, "recv": self.recv_size},
            Tensor: {"send": self.send_tensor, "recv": self.recv_tensor},
            MicroLossTensor: {
                "send": self.send_tensor,
                "recv": self.recv_tensor,
            },
        }

        self.TORCH_ID_TO_DTYPE = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]

        self.ID_TO_DTYPE = list(self.INSTRUCTIONS.keys())
        self.DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(self.ID_TO_DTYPE)}
        self.TORCH_DTYPE_TO_ID = {
            dtype: idx for idx, dtype in enumerate(self.TORCH_ID_TO_DTYPE)
        }

    def ring_exchange_forward(self, value):
        """
        Exchange values lower to higher stage

        e.g. 0 -> 1 -> 2 -> 3
        """
        if not self.mpu.is_pipeline_first_stage():
            value = self.recv(self.stage_id - 1)

        if not self.mpu.is_pipeline_last_stage():
            self.send(value, self.stage_id + 1)

        return value

    def ring_exchange_backward(self, value):
        """
        Exchange values higher to lower stage

        e.g. 3 -> 2 -> 1 -> 0
        """
        if not self.mpu.is_pipeline_last_stage():
            value = self.recv(self.stage_id + 1)

        if not self.mpu.is_pipeline_first_stage():
            self.send(value, self.stage_id - 1)

        return value

    def send(self, value, recv_stage):
        """Send data that have every type to other stage"""
        _type = type(value)
        assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"

        return self.INSTRUCTIONS[_type]["send"](
            value, recv_stage=recv_stage, send_type=True
        )

    def recv(self, send_stage):
        """Receive data that have every type from other stage"""
        _type = self.INSTRUCTIONS[type]["recv"](send_stage)
        return self.INSTRUCTIONS[_type]["recv"](send_stage)

    def send_type(self, _type, recv_stage, send_type=False):
        """Send type to other stage"""
        assert send_type is False, "to send ``type``, we don't need to send type."

        send_type = torch.tensor(
            [self.DTYPE_TO_ID[_type]],
            dtype=torch.long,
            device=self.device,
        )

        assert self.mpu.p2p(
            send_type, self.stage_id, recv_stage
        ), f"Communication failed: send_type_{self.stage_id}_to_{recv_stage}"

    def recv_type(self, send_stage):
        """Receive type from other stage"""
        recv_type = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_type, send_stage, self.stage_id
        ), f"Communication failed: recv_type_{send_stage}_to_{self.stage_id}"
        return self.ID_TO_DTYPE[recv_type.item()]

    def send_none(self, none, recv_stage, send_type=False):
        """Send None to other stage"""
        assert none is None, f"wrong type: {none} must be {NoneType} type"

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](NoneType, recv_stage)

    def recv_none(self, send_stage):
        """Receive None from other stage"""
        return None

    def send_str(self, _str, recv_stage, send_type=False):
        """Send string to other stage, note we only support ascii string now"""
        assert isinstance(_str, str), f"wrong type: {_str} must be {str} type."
        assert all(ord(c) < 128 for c in _str), "send string must be ascii string."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](str, recv_stage)

        send_len_string = torch.tensor(
            [len(_str)],
            dtype=torch.long,
            device=self.device,
        )

        assert self.mpu.p2p(
            send_len_string, self.stage_id, recv_stage
        ), f"Communication failed: send_str_{self.stage_id}_to_{recv_stage}"

        send_string = torch.tensor(
            [ord(s) for s in _str], dtype=torch.long, device=self.device
        )

        assert self.mpu.p2p(send_string, self.stage_id, recv_stage)

    def recv_str(self, send_stage):
        """Receive string from other stage"""
        recv_len_string = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(recv_len_string, send_stage, self.stage_id)
        recv_len_string = recv_len_string.item()

        recv_string = torch.tensor(
            [0] * recv_len_string,
            dtype=torch.long,
            device=self.device,
        )
        assert self.mpu.p2p(
            recv_string, send_stage, self.stage_id
        ), f"Communication failed: recv_type_{send_stage}_to_{self.stage_id}"
        return "".join([chr(s) for s in recv_string])

    def send_bool(self, _bool, recv_stage, send_type=False):
        """Send boolean to other stage"""
        assert isinstance(_bool, bool), f"wrong type: {_bool} must be {bool} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](bool, recv_stage)

        send_boolean = torch.tensor(
            [1 if _bool else 0], dtype=torch.long, device=self.device
        )

        assert self.mpu.p2p(
            send_boolean, self.stage_id, recv_stage
        ), f"Communication failed: send_bool_{self.stage_id}_to_{recv_stage}"

    def recv_bool(self, send_stage):
        """Receive boolean from other stage"""
        recv_boolean = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_boolean, send_stage, self.stage_id
        ), f"Communication failed: recv_bool_{send_stage}_to_{self.stage_id}"
        recv_boolean = recv_boolean.item()

        if recv_boolean == 0:
            return False
        elif recv_boolean == 1:
            return True
        else:
            raise ValueError(
                f"Wrong value for boolean. only 0 or 1 can be supported. "
                f"but your input is {recv_boolean}"
            )

    def send_int(self, _int, recv_stage, send_type=False):
        """Send int to other stage"""
        assert isinstance(_int, int), f"wrong type: {_int} must be {int} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](int, recv_stage)

        send_int = torch.tensor([_int], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            send_int, self.stage_id, recv_stage
        ), f"Communication failed: send_int_{self.stage_id}_to_{recv_stage}"

    def recv_int(self, send_stage):
        """Receive int from other stage"""
        recv_int = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            recv_int, send_stage, self.stage_id
        ), f"Communication failed: recv_int_{send_stage}_to_{self.stage_id}"
        return recv_int.item()

    def send_float(self, _float, recv_stage, send_type=False):
        """Send float to other stage"""
        assert isinstance(_float, float), f"wrong type: {_float} must be {float} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](float, recv_stage)

        send_float = torch.tensor([_float], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_float, self.stage_id, recv_stage
        ), f"Communication failed: send_float_{self.stage_id}_to_{recv_stage}"

    def recv_float(self, send_stage):
        """Receive float from other stage"""
        recv_float = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_float, send_stage, self.stage_id
        ), f"Communication failed: recv_float_{send_stage}_to_{self.stage_id}"
        return recv_float.item()

    def send_complex(self, _complex, recv_stage, send_type=False):
        """Send complex to other stage"""
        assert isinstance(
            _complex, complex
        ), f"wrong type: {_complex} must be {complex} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](complex, recv_stage)

        send_real = torch.tensor([_complex.real], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_real, self.stage_id, recv_stage
        ), f"Communication failed: send_complex_{self.stage_id}_to_{recv_stage}"
        send_imag = torch.tensor([_complex.imag], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            send_imag, self.stage_id, recv_stage
        ), f"Communication failed: send_complex_{self.stage_id}_to_{recv_stage}"

    def recv_complex(self, send_stage):
        """Receive complex from other stage"""
        recv_real = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_real, send_stage, self.stage_id
        ), f"Communication failed: recv_complex_{send_stage}_to_{self.stage_id}"
        recv_imag = torch.tensor([0.0], dtype=torch.float, device=self.device)
        assert self.mpu.p2p(
            recv_imag, send_stage, self.stage_id
        ), f"Communication failed: recv_complex_{send_stage}_to_{self.stage_id}"

        return complex(recv_real.item(), recv_imag.item())

    def send_tensor(self, _tensor, recv_stage, send_type=False):
        """Send tensor to other stage"""
        assert isinstance(
            _tensor, Tensor
        ), f"wrong type: {_tensor} must be {Tensor} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](Tensor, recv_stage)

        # dtype is float32 or float16, ... (type of element)
        _dtype = self.TORCH_DTYPE_TO_ID[_tensor.dtype]
        _dtype = torch.tensor(_dtype, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _dtype, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _requires_grad = torch.tensor(
            1 if _tensor.requires_grad else 0,
            dtype=torch.long,
            device=self.device,
        )
        assert self.mpu.p2p(
            _requires_grad, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _ndims = len(_tensor.size())
        _ndims = torch.tensor(_ndims, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _ndims, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        _shape = list(_tensor.size())
        _shape = torch.tensor(_shape, dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _shape, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

        if _tensor.dtype == torch.bool:
            _tensor = _tensor.half()

        if not _tensor.is_contiguous():
            _tensor = _tensor.contiguous()

        assert self.mpu.p2p(
            _tensor, self.stage_id, recv_stage
        ), f"Communication failed: send_tensor_{self.stage_id}_to_{recv_stage}"

    def recv_tensor(self, send_stage):
        """Receive tensor from other stage"""
        _dtype = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _dtype, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _dtype = self.TORCH_ID_TO_DTYPE[_dtype.item()]

        _requires_grad = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _requires_grad, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _requires_grad = True if _requires_grad.item() == 1 else False

        _ndims = torch.tensor([0], dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _ndims, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _shape = torch.tensor([0] * _ndims.item(), dtype=torch.long, device=self.device)
        assert self.mpu.p2p(
            _shape, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"
        _shape = tuple(_shape.tolist())

        if _dtype == torch.bool:
            __dtype = torch.half
        else:
            __dtype = _dtype

        recv_tensor = torch.zeros(size=_shape, dtype=__dtype, device=self.device)
        recv_tensor.requires_grad = _requires_grad and recv_tensor.is_floating_point()

        assert self.mpu.p2p(
            recv_tensor, send_stage, self.stage_id
        ), f"Communication failed: recv_tensor_{send_stage}_to_{self.stage_id}"

        if _dtype == torch.bool:
            recv_tensor = recv_tensor.bool()

        return recv_tensor

    def send_list(self, _list, recv_stage, send_type=False):
        """Send list to other stage"""
        assert isinstance(_list, list), f"wrong type: {_list} must be {list} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](list, recv_stage)

        list_len = len(_list)
        self.send_int(list_len, recv_stage)

        for item in _list:
            _type = type(item)
            assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"
            self.INSTRUCTIONS[_type]["send"](item, recv_stage, send_type=True)

    def recv_list(self, send_stage):
        """Receive list from other stage"""
        len_list = self.recv_int(send_stage)
        output_list = []

        for _ in range(len_list):
            _type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _recv = self.INSTRUCTIONS[_type]["recv"](send_stage)
            output_list.append(_recv)

        return output_list

    def send_set(self, _set, recv_stage, send_type=False):
        """Send set to other stage"""
        assert isinstance(_set, set), f"wrong type: {_set} must be {set} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](set, recv_stage)

        self.send_list(list(_set), recv_stage, False)

    def recv_set(self, send_stage):
        """Receive set from other stage"""
        return set(self.recv_list(send_stage))

    def send_tuple(self, _tuple, recv_stage, send_type=False):
        """Send tuple to other stage"""
        assert isinstance(_tuple, tuple), f"wrong type: {_tuple} must be {tuple} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](tuple, recv_stage)

        self.send_list(list(_tuple), recv_stage, send_type=False)

    def recv_tuple(self, send_stage):
        """Receive tuple from other stage"""
        return tuple(self.recv_list(send_stage))

    def send_size(self, _size, recv_stage, send_type=False):
        """Send torch.Size to other stage"""
        assert isinstance(_size, Size), f"wrong type: {_size} must be {Size} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](Size, recv_stage)

        self.send_list(list(_size), recv_stage, send_type=False)

    def recv_size(self, send_stage):
        """Receive torch.Size from other stage"""
        return Size(self.recv_list(send_stage))

    def send_dict(self, _dict, recv_stage, send_type=False):
        """Send dict to other stage"""
        assert isinstance(_dict, dict), f"wrong type: {_dict} must be {dict} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](dict, recv_stage)

        dict_len = len(_dict)
        self.send_int(dict_len, recv_stage)

        for key, val in _dict.items():
            _type_key, _type_val = type(key), type(val)
            assert _type_key in self.ID_TO_DTYPE, f"unsupported type: {_type_key}"
            assert _type_val in self.ID_TO_DTYPE, f"unsupported type: {_type_val}"
            self.INSTRUCTIONS[_type_key]["send"](key, recv_stage, send_type=True)
            self.INSTRUCTIONS[_type_val]["send"](val, recv_stage, send_type=True)

    def recv_dict(self, send_stage):
        """Receive dict from other stage"""
        len_dict = self.recv_int(send_stage)
        output_dict = {}

        for _ in range(len_dict):
            _key_type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _key_recv = self.INSTRUCTIONS[_key_type]["recv"](send_stage)
            _val_type = self.INSTRUCTIONS[type]["recv"](send_stage)
            _val_recv = self.INSTRUCTIONS[_val_type]["recv"](send_stage)
            output_dict[_key_recv] = _val_recv

        return output_dict
