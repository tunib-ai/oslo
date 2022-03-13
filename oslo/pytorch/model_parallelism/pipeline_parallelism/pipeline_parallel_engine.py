import time
from math import sqrt

import psutil
import torch
import torch.distributed as dist
from anytree import Node

from oslo.pytorch.model_parallelism.network.broadcaster import Broadcaster
from oslo.pytorch.model_parallelism.utils.extensions import get_parameter_dtype
from oslo.pytorch.utils.huggingface import is_huggingface_model


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
    Pipeline parallel engine based on Amazon Sagemaker.
    """

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
        self.memory_computation_balance_factor = memory_computation_balance_factor

    def parallelize(self):
        # partition model parameters
        partitioner = ModelPartitioner(
            model=self.model,
            mpu=self.mpu,
            tracing_inputs=self.tracing_inputs,
            memory_computation_balance_factor=self.memory_computation_balance_factor,
        )
        partitioner.partition()


class PipelineDeparallelEngine(object):
    pass


class P2PCommunicator(object):
    def __init__(self, model):
        self.model = model
        self.broadcaster = Broadcaster()


class ModelPartitioner(object):
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
        self.model = model
        self.mpu = mpu
        self.tracing_inputs = tracing_inputs
        self.memory_computation_balance_factor = memory_computation_balance_factor

        self.visited = {}
        self.bfs = {}
        self.partition_memoization = {}

    def partition(self):
        # 1. construct tree
        self.root_node = Node(
            name=self.model.__class__.__qualname__,
            parent=None,
            modules=[self.model],
            parameters=self._get_parameters(self.model),
            execution_order=0,
            cost=1.0,
        )
        self._construct_tree(self.root_node, self.root_node.name)

        # 2. compute the partitioning cost
        cost_estimator = PartitioningCostEstimator(
            root_node=self.root_node,
            alpha=self.memory_computation_balance_factor,
            tracing_inputs=self.tracing_inputs,
        )
        cost_estimator.compute_cost()

        # 3. partition tree
        self._tree_partitioning()

        # 4. set device to parameters and buffers
        for node in dfs(self.root_node):
            for parameter in node.modules[0].parameters():
                setattr(parameter, "pp_rank", node.device)
            for buffer in node.modules[0].buffers():
                setattr(buffer, "pp_rank", node.device)

    @staticmethod
    def _get_parameters(module, to_list=True):
        parameters = module.parameters()
        parameters = list(parameters) if to_list else tuple(parameters)
        return parameters

    def _construct_tree(self, node, parent_name):
        for name, child in node.modules[0].named_children():
            name = f"{parent_name}.{name}" if parent_name != "ROOT" else name
            parameters = self._get_parameters(child, to_list=False)

            if len(parameters) != 0:
                visited_node = self.visited.get(parameters, None)
                if visited_node and parent_name not in visited_node.name:
                    child_node = self.visited[parameters]
                    child_node.modules.append(child)
                    child_node.name = ",".join([child_node.name, name])
                    setattr(parameters, "tied_parameters", True)
                else:
                    child_node = Node(
                        name=name,
                        parent=node,
                        modules=[child],
                        parameters=list(parameters),
                    )
                    self.visited[parameters] = child_node
                self._construct_tree(child_node, name)

    @staticmethod
    def _partition_segments(sequence, k):
        n = len(sequence)
        if k >= n:
            return [[node] for node in sequence]
        sum = [0] * (n + 1)
        dp = [[0 for _ in range(n + 1)] for _ in range(2)]
        tb = [[[] for _ in range(n + 1)] for _ in range(2)]

        for a in range(0, n):
            sum[a + 1] = sequence[a].cost + sum[a]
            dp[0][a + 1] = float("inf")
            dp[1][a + 1] = float("inf")

        for a in range(1, k + 1):
            for b in range(a + 1, n + 1):
                for c in range(a - 1, b):
                    if max(dp[(a - 1) % 2][c], sum[b] - sum[c]) < dp[a % 2][b]:
                        dp[a % 2][b] = max(dp[(a - 1) % 2][c], sum[b] - sum[c])
                        tb[a % 2][b] = tb[(a - 1) % 2][c] + [c]

        starts = tb[k % 2][n]
        ends = starts[1:] + [n]
        return [sequence[s:e] for s, e in zip(starts, ends)]

    # Algorithm 1
    def _tree_partitioning(self):
        # The algorithm starts with a set of virtual devices
        # P(r) = {0, 1, . . . , D − 1} for the root node r
        initial_partition = [
            p for p in range(self.mpu.get_pipeline_parallel_world_size())
        ]
        # P(n)
        setattr(self.root_node, "device_cands", initial_partition)
        # d(n)
        setattr(self.root_node, "device", self.root_node.device_cands[0])

        for node in bfs(self.root_node, self.bfs):
            p_n = node.device_cands
            setattr(node, "device", p_n[0])  # d(n)

            q_n = node.children
            if len(q_n) > 0:
                if len(p_n) > 1:
                    P = self._partition(p_n, q_n)
                    for q, device_cands in zip(q_n, P):
                        setattr(q, "device_cands", device_cands)
                else:
                    for q in q_n:
                        setattr(q, "device_cands", [node.device_cands[0]])

    # Algorithm 2
    def _partition(self, p_n, q_n):
        # 1. Find l segments {Si}_0≤i≤l−1 by solving (2) using the recursion (3).
        nodes = tuple(q_n)
        partition = self._partition_segments(nodes, len(p_n))

        # 2. Compute segment costs
        segments = [
            Node(name=i, segment=segment, segment_cost=sum(n.cost for n in segment))
            for i, segment in enumerate(partition)
        ]

        # 3, Compute segment allocations
        P = self._d_hondt(p_n, segments)
        S = [segment.segment for segment in segments]

        for i in range(len(S)):
            if len(P[i]) == 0:
                for node in S[i]:
                    setattr(node, "device_cands", [p_n[0]])
            elif len(S[i]) == 1 or len(P[i]) == 1:
                for node in S[i]:
                    setattr(node, "device_cands", P[i])
            else:
                self._partition(P[i], S[i])

        return [q.device_cands for q in q_n]

    # Algorithm 3
    @staticmethod
    def _d_hondt(p_n, segments):
        s = 1
        P = {i: [] for i in range(len(segments))}
        Q = {i: seg.segment_cost for i, seg in enumerate(segments)}
        n_children = {
            i: sum([len(list(n.modules[0].modules())) for n in seg.segment])
            for i, seg in enumerate(segments)
        }
        sqrt_mean_children = sqrt(sum(n_children.values()) / len(n_children.values()))

        for p in p_n:
            k = max(Q, key=Q.get)
            P[k].append(p)
            if n_children[k] <= sqrt_mean_children:
                # This part is different from the paper. For a small model,
                # the embedding layer will have a significantly large size
                # Therefore, too many devices are allocated for embedding.
                # To solve this problem, we reduce the segment cost to zero
                # if the number of submodules in a node is less than the square
                # root of the mean of the total number of submodules.
                Q[k] = 0
            else:
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
        if self.orig_gradient_checkpointing_status is False:
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


def main():
    from transformers import GPT2Model

    from oslo.pytorch.model_parallelism.model_parallel_engine import (
        ModelParallelEngine,
    )
    from oslo.pytorch.model_parallelism.network.mpu import MPU

    mpu = MPU(1, 4)
    model = GPT2Model.from_pretrained("gpt2")
    pp = ModelParallelEngine(
        model=model,
        mpu=mpu,
        tp_mapping=None,
        memory_computation_balance_factor=1.0,
        tracing_inputs=None,
    )
    pp.parallelize()


if __name__ == "__main__":
    main()
