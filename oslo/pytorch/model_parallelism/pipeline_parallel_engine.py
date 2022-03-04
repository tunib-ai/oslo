import time
from dataclasses import dataclass
from typing import Tuple

import psutil
import torch
import torch.nn as nn

from oslo.pytorch.utils.huggingface import is_huggingface_model


@dataclass
class Segment(object):
    # ``S`` in the paper.
    modules: Tuple[nn.Module]

    @property
    def cost(self):
        return sum([m.cost for m in self.modules])


@dataclass
class Partition(object):
    # ``P`` in the paper.
    segments: Tuple[Segment]

    @property
    def cost(self):
        return sum([m.cost for m in self.segments])


class PipelineParallelEngine(object):
    """
    For more information of the implementation, see the following paper.

    References:
        Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training
        https://arxiv.org/abs/2111.05972
    """

    def __init__(
        self,
        model,
        mpu,
        tracing_inputs=None,
        memory_computation_balance_factor=1.0,
    ):
        assert hasattr(
            model, "get_input_embeddings"
        ), "model must have method named ``get_input_embeddings()``."

        self.model = model
        self.mpu = mpu
        self.tracing_inputs = tracing_inputs

        # 1. compute the partitioning cost
        cost_estimator = PartitioningCostEstimator(
            root_node=self.model,
            alpha=memory_computation_balance_factor,
            tracing_inputs=tracing_inputs,
        )
        cost_estimator.compute_cost()

        # 2. Do partitioning
        self.initialize_partition()

    def initialize_partition(self):
        # The algorithm starts with a set of virtual devices
        # P(r) = {0, 1, . . . , D − 1} for the root node r
        initial_partition = [
            p for p in range(self.mpu.get_pipeline_parallel_world_size())
        ]
        # P(n)
        setattr(self.model, "oslo_pp_device_cands", initial_partition)
        # d(n)
        setattr(self.model, "oslo_pp_device", self.model.oslo_pp_device_cands[0])

        self.make_segments(self.model)

    def make_segments(self, node):
        children = sorted(
            node.children(),
            key=lambda x: x.oslo_execution_order,
        )

        for i, child in enumerate(children):
            if torch.distributed.get_rank() == 0:
                print(child.oslo_execution_order)

    def dhondt(self, node):
        pass


class PartitioningCostEstimator(object):
    """
    Partitioning cost estimator

    1. computation cost: supports only the cpu time estimating in this version.
    2. memory cost: computes memory cost via the number of parameters of the module.
    """

    def __init__(self, root_node, alpha, tracing_inputs):
        self.root_node = root_node
        self.alpha = alpha
        self.tracing_inputs = tracing_inputs
        self.node_order = 0

        self.hooks = []
        if is_huggingface_model(root_node):
            # enable gradient checkpointing for memory safer tracing.
            self.orig_gradient_checkpointing_status = (
                self.root_node.is_gradient_checkpointing
            )
            if self.root_node.supports_gradient_checkpointing:
                self.root_node.gradient_checkpointing_enable()
        else:
            self.orig_gradient_checkpointing_status = None

        # prevent tracing for very large model.
        if self.alpha < 1.0:
            if not self._is_available_tracing(self.root_node):
                print(
                    "This model is too large to trace on the CPU."
                    "turn off computation cost estimating."
                )
                self.use_computation_cost = False
            else:
                if tracing_inputs is None and not is_huggingface_model(self.root_node):
                    raise ValueError(
                        "`tracing_inputs` must not be None "
                        "if the model is not Hugging Face Transformers model"
                    )
                self._trace_computation_cost(tracing_inputs)

    @staticmethod
    def _is_available_tracing(module):
        elem_size = torch.zeros(1, dtype=module.dtype).element_size()
        model_memory_size = (
            sum(p.numel() for p in module.parameters() if p.requires_grad) * elem_size
        )
        available_memory_size = psutil.virtual_memory().available
        return available_memory_size > model_memory_size * 2
        # multiply by 2 to consider act memory for safer tracing.

    def _add_computation_cost_hooks(self, node):
        # TODO: The time unit is not mentioned in the paper.
        # I sent an email to author of paper. but he didn't reply :(

        def pre_hook(*args, **kwargs):
            setattr(node, "execution_time_before_tracing", time.time())
            setattr(node, "oslo_execution_order", self.node_order)
            self.node_order += 1

        def post_hook(*args, **kwargs):
            setattr(
                node,
                "oslo_pp_computation_cost",
                time.time() - getattr(node, "execution_time_before_tracing"),
            )
            delattr(node, "execution_time_before_tracing")

        self.hooks.append(
            {
                "pre_hook": node.register_forward_pre_hook(pre_hook),
                "post_hook": node.register_forward_hook(post_hook),
            }
        )

        for name, child in node.named_children():
            self._add_computation_cost_hooks(child)

    def _trace_computation_cost(self, tracing_inputs):
        # 1. tracing the model
        with torch.no_grad():
            if tracing_inputs is None:
                tracing_inputs = self.root_node.dummy_inputs
                tracing_inputs["use_cache"] = False  # for checkpointing

            self._add_computation_cost_hooks(self.root_node)
            self.root_node(**tracing_inputs)

        # 2. removing hooks
        for hooks in self.hooks:
            hooks["pre_hook"].remove()
            hooks["post_hook"].remove()

        # 3. turn off gradient checkpointing
        if self.orig_gradient_checkpointing_status is False:
            self.root_node.gradient_checkpointing_disable()

    def _compute_node_cost(self, node):
        # 1. compute memory cost
        memory_cost = sum(p.numel() for p in node.parameters() if p.requires_grad)

        # 2. compute computation cost if available
        computation_cost = (
            node.oslo_pp_computation_cost
            if hasattr(node, "oslo_pp_computation_cost")
            else 0.0
        )

        # 3. compute total cost
        total_cost = (self.alpha * memory_cost) + ((1 - self.alpha) * computation_cost)

        setattr(node, "oslo_pp_unnormalized_cost", total_cost)

    def _compute_cost(self, node):
        if not hasattr(self.root_node, "oslo_pp_unnormalized_cost"):
            # 1. compute cost for root node
            self._compute_node_cost(self.root_node)
        else:
            # 2. compute cost for children nodes
            self._compute_node_cost(node)

        # 3. do recursion
        for child in node.children():
            self._compute_cost(child)

    def _normalize_cost(self, node):
        if not hasattr(self.root_node, "oslo_pp_cost"):
            # 1. normalize cost for root node
            setattr(self.root_node, "oslo_pp_cost", 1.0)
        else:
            # 2. normalize cost for children nodes
            root_cost = getattr(self.root_node, "oslo_pp_unnormalized_cost")
            node_cost = getattr(node, "oslo_pp_unnormalized_cost")
            setattr(node, "oslo_pp_cost", node_cost / root_cost)
            delattr(node, "oslo_pp_unnormalized_cost")

        # 3. do recursion
        for child in node.children():
            self._normalize_cost(child)

    def compute_cost(self):
        # 1. compute cost
        self._compute_cost(self.root_node)

        # 2. normalize cost
        self._normalize_cost(self.root_node)
        delattr(self.root_node, "oslo_pp_unnormalized_cost")

        previous_order, additional_order_count = 0, 0
        for name, module in self.root_node.named_modules():
            if not hasattr(module, "oslo_execution_order"):
                # for module list
                setattr(module, "oslo_execution_order", previous_order + 1)
                additional_order_count += 1
            else:
                setattr(
                    module,
                    "oslo_execution_order",
                    module.oslo_execution_order + additional_order_count,
                )

            previous_order = module.oslo_execution_order


if __name__ == "__main__":
    from transformers import GPT2Model

    from oslo.pytorch.model_parallelism.network.mpu import MPU

    model = GPT2Model.from_pretrained("gpt2")
    mpu = MPU(1, 2)
    pp = PipelineParallelEngine(model, mpu)
