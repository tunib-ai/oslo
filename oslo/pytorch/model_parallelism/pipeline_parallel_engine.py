import sys
import time

import psutil
import torch
import torch.distributed as dist
from anytree import Node

from oslo.pytorch.model_parallelism.utils.extensions import get_parameter_dtype
from oslo.pytorch.utils.huggingface import is_huggingface_model

PARTITIONS_MEMOIZATION = {}


def dfs(node, bfs_dict=None):
    yield node
    if bfs_dict is not None:
        if node.depth in bfs_dict:
            bfs_dict[node.depth].append(node)
        else:
            bfs_dict[node.depth] = [node]

    for child in node.children:
        for c in dfs(child, bfs_dict):
            yield c


def bfs(node, bfs_dict=None):
    if bfs_dict is None:
        bfs_dict = {}
    if len(bfs_dict) == 0:
        list(dfs(node, bfs_dict))
    for nodes in bfs_dict.values():
        for node in nodes:
            yield node


class PipelineParallelEngine(object):
    """
    For more information of the implementation, see the following paper.

    References:
        Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training
        https://arxiv.org/abs/2111.05972
    """

    __PARTITIONING_MEMO__ = {}

    def __init__(
        self,
        model,
        mpu,
        tracing_inputs=None,
        memory_computation_balance_factor=1.0,
    ):
        self.model = model
        self.mpu = mpu
        self.tracing_inputs = tracing_inputs
        self.visited = {}
        self.bfs = {}
        self.partition_memoization = {}

        # 1. construct tree
        self.root_node = Node(
            name="ROOT",
            parent=None,
            modules=[self.model],
            parameters=self.get_parameters(self.model),
            execution_order=0,
            cost=1.0,
        )
        self.construct_tree(self.root_node)

        # 2. compute the partitioning cost
        cost_estimator = PartitioningCostEstimator(
            root_node=self.root_node,
            alpha=memory_computation_balance_factor,
            tracing_inputs=tracing_inputs,
        )
        cost_estimator.compute_cost()

        # 3. partition tree
        self.tree_partitioning()

    @staticmethod
    def get_parameters(module, to_list=True):
        parameters = module.parameters()
        parameters = list(parameters) if to_list else tuple(parameters)
        return parameters

    def construct_tree(self, node):
        for name, child in node.modules[0].named_children():
            parameters = self.get_parameters(child, to_list=False)
            if len(parameters) != 0:
                if parameters in self.visited:
                    child_node = self.visited[parameters]
                    child_node.modules.append(child)
                    child_node.name = ",".join([child_node.name, name])
                else:
                    child_node = Node(
                        name=name,
                        parent=node,
                        modules=[child],
                        parameters=list(parameters),
                    )
                    self.visited[parameters] = child_node
                self.construct_tree(child_node)

    def tree_partitioning(self):
        # The algorithm starts with a set of virtual devices
        # P(r) = {0, 1, . . . , D − 1} for the root node r
        initial_partition = [
            p for p in range(self.mpu.get_pipeline_parallel_world_size())
        ]
        # P(n)
        setattr(self.root_node, "device_cands", initial_partition)
        # d(n)
        setattr(self.root_node, "device", self.root_node.device_cands[0])
        self._tree_partitioning()

    def _tree_partitioning(self):
        for node in bfs(self.root_node, self.bfs):
            p_n = node.device_cands
            setattr(node, "device", p_n[0])  # d(n)

            q_n = node.children
            if len(q_n) > 0:
                if len(p_n) > 1:
                    P = self.partition(p_n, q_n)
                    for q, device_cands in zip(q_n, P):
                        setattr(q, "device_cands", device_cands)
                else:
                    for q in q_n:
                        setattr(q, "device_cands", [node.device_cands[0]])

    def _partition(self, c, k, nodes):
        key = k, tuple(node.cost for node in nodes)
        result = self.__PARTITIONING_MEMO__.get(key, None)
        if result is not None:
            return result

        i = len(nodes)
        if k == 1:
            c[(k, i)] = sum(n.cost for n in nodes)
            result = (nodes,)
        elif k >= len(nodes):
            c[(k, i)] = max(n.cost for n in nodes)
            result = tuple((s,) for s in nodes)
        else:
            first, tail = nodes[:1], nodes[1:]
            seg = self._partition(c, k, tail)
            candidates = [(first, *self._partition(c, k - 1, tail))] + [
                seg[:j] + (first + seg[j],) + seg[j + 1 :] for j in range(len(seg))
            ]

            result = tuple()
            for partition in candidates:
                cost = max([sum(j.cost for j in S) for S in partition])
                if c[(k, i)] >= cost:
                    c[(k, i)] = cost
                    result = partition

        self.__PARTITIONING_MEMO__[key] = result
        return result

    def partition(self, p_n, q_n):
        # Find l segments {Si}_0≤i≤l−1 by solving (2) using the recursion (3).
        c = {
            (k, i): sys.maxsize
            for k in range(len(p_n) + 1)
            for i in range(len(q_n) + 1)
        }
        nodes = tuple(q_n)
        partition = self._partition(c, len(p_n), nodes)  # choose l = |P(n)|
        # Compute segment costs
        segments = [
            Node(name=i, segment=segment, segment_cost=sum(n.cost for n in segment))
            for i, segment in enumerate(partition)
        ]

        # Compute segment allocations
        P = self._d_hondt(p_n, segments)
        S = [segment.segment for segment in segments]

        for i in range(len(P)):
            if len(P[i]) == 0:
                for node in S[i]:
                    setattr(node, "device_cands", [p_n[0]])
            elif len(S[i]) == 1 or len(P[i]) == 1:
                for node in S[i]:
                    setattr(node, "device_cands", P[i])
            else:
                self.partition(P[i], S[i])

        return [q.device_cands for q in q_n]

    def _d_hondt(self, p_n, segments):
        s = 1
        P = {i: [] for i in range(len(p_n))}
        Q = {i: seg.segment_cost for i, seg in enumerate(segments)}

        for p in p_n:
            k = max(Q, key=Q.get)
            P[k].append(p)
            Q[k] /= s + 1
            s += 1

        return list(P.values())


class PartitioningCostEstimator(object):
    """
    Partitioning cost estimator

    1. computation cost: supports only the cpu time estimating in this version.
    2. memory cost: computes memory cost via the number of parameters of the module.
    """

    def __init__(self, root_node, alpha, tracing_inputs):
        self.root_node = root_node
        self.model = self.root_node.modules[0]
        self.alpha = alpha
        self.tracing_inputs = tracing_inputs
        self.node_order = 0

        self.hooks = []
        if is_huggingface_model(self.model):
            # enable gradient checkpointing for memory safer tracing.
            self.orig_gradient_checkpointing_status = (
                self.model.is_gradient_checkpointing
            )
            if self.model.supports_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.orig_gradient_checkpointing_status = None

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
                tracing_inputs["use_cache"] = False  # for checkpointing

            for node in dfs(self.root_node):
                self.hooks.append(self._add_computation_cost_hooks(node))

            self.model(**tracing_inputs)

        # 2. removing hooks
        for hooks in self.hooks:
            hooks["pre_hook"].remove()
            hooks["post_hook"].remove()

        # 3. turn off gradient checkpointing
        if self.orig_gradient_checkpointing_status is True:
            self.model.gradient_checkpointing_disable()

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


# if __name__ == "__main__":
#     from transformers import GPT2Model
#
#     from oslo.pytorch.model_parallelism.network.mpu import MPU
#
#     mpu = MPU(1, 4)
#     model = GPT2Model.from_pretrained("gpt2")
#     pp = PipelineParallelEngine(model, mpu, memory_computation_balance_factor=0.2)
