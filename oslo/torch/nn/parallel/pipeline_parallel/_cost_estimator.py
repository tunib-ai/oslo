import time
from typing import Any, Dict

import psutil
import torch
import torch.distributed as dist
from anytree import Node

from oslo.torch.nn.parallel.pipeline_parallel._utils import dfs
from oslo.torch.nn.parallel.utils import (
    get_parameter_dtype,
    is_huggingface_model,
)


class PartitioningCostEstimator(object):
    """
    Partitioning cost estimator

    Args:
        root_node (Node): Root node of model
        alpha (float): memory computation balance factor
        tracing_inputs (Dict[str, Any]): tracing input dictionary, will be input as **kwargs to model.

    Notes:
        1. computation cost: supports only the cpu time estimating in this version.
        2. memory cost: computes memory cost via the number of parameters of the module.

    References:
        Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training
        https://arxiv.org/abs/2111.05972
    """

    def __init__(
        self,
        root_node: Node,
        alpha: float,
        tracing_inputs: Dict[str, Any],
    ):
        self.root_node = root_node
        self.model = self.root_node.modules[0]
        self.tracing_inputs = tracing_inputs
        self.alpha = alpha
        self.node_order = 0
        self.hooks = []

        # prevent tracing for very large model.
        if self.alpha < 1.0:
            if not self._is_available_tracing(self.model):
                print(
                    "This model is too large to trace on the CPU."
                    "turn off computation cost estimating."
                )
                self.use_computation_cost = False
            else:
                if tracing_inputs is None and not is_huggingface_model(self.model):
                    raise ValueError(
                        "`tracing_inputs` must not be None "
                        "if the model is not Hugging Face Transformers model"
                    )
                self._trace_computation_cost(tracing_inputs)

    @staticmethod
    def _is_available_tracing(module):
        elem_size = torch.zeros(1, dtype=get_parameter_dtype(module)).element_size()
        model_memory_size = (
            sum(p.numel() for p in module.parameters() if p.requires_grad) * elem_size
        )
        available_memory_size = psutil.virtual_memory().available
        return available_memory_size > model_memory_size * dist.get_world_size() * 1.5
        # multiply 1.5 for safer tracing.

    def _add_computation_cost_hooks(self, node):
        def pre_hook(*args, **kwargs):
            setattr(node, "execution_time_before_tracing", time.time())
            setattr(node, "execution_order", self.node_order)
            self.node_order += 1

        def post_hook(*args, **kwargs):
            before = node.execution_time_before_tracing
            setattr(node, "computation_cost", time.time() - before)
            delattr(node, "execution_time_before_tracing")

        pre_hook = node.modules[0].register_forward_pre_hook(pre_hook)
        post_hook = node.modules[0].register_forward_hook(post_hook)

        return {"pre_hook": pre_hook, "post_hook": post_hook}

    def _trace_computation_cost(self, tracing_inputs):
        # 1. tracing the model
        with torch.no_grad():
            if tracing_inputs is None:
                tracing_inputs = self.model.dummy_inputs
                tracing_inputs["use_cache"] = False

            for node in dfs(self.root_node):
                self.hooks.append(self._add_computation_cost_hooks(node))

            self.model(**tracing_inputs)

        # 2. removing hooks
        for hooks in self.hooks:
            hooks["pre_hook"].remove()
            hooks["post_hook"].remove()

    def _compute_cost(self):
        for node in dfs(self.root_node):
            # 1. compute memory cost
            memory_cost = sum(p.numel() for p in node.parameters if p.requires_grad)

            # 2. compute computation cost if available
            computation_cost = (
                node.computation_cost if hasattr(node, "computation_cost") else 0.0
            ) * 1000  # milliseconds (It is set arbitrarily because it is not mentioned in the paper.)

            # 3. compute total cost
            total_cost = (self.alpha * memory_cost) + (
                (1 - self.alpha) * computation_cost
            )
            setattr(node, "unnormalized_cost", total_cost)

    def _normalize_cost(self):
        for node in dfs(self.root_node):
            root_cost = self.root_node.unnormalized_cost
            node_cost = node.unnormalized_cost
            setattr(node, "cost", node_cost / root_cost)

            if node is not self.root_node:
                delattr(node, "unnormalized_cost")

    def _fix_execution_order_for_module_list(self):
        prev_order, add_order = 0, 0
        for node in dfs(self.root_node):
            if not hasattr(node, "execution_order"):
                setattr(node, "execution_order", prev_order + 1)
                add_order += 1
            else:
                node.execution_order += add_order

            prev_order = node.execution_order

    def _sort_children_by_execution_order(self):
        for node in dfs(self.root_node):
            node.children = sorted(
                node.children, key=lambda child: child.execution_order
            )

    def compute_cost(self):
        # 1. compute cost
        self._compute_cost()

        # 2. normalize cost
        self._normalize_cost()
        delattr(self.root_node, "unnormalized_cost")

        # 3. fix execution order for module list
        self._fix_execution_order_for_module_list()

        # 4. sort children by execution order
        self._sort_children_by_execution_order()
