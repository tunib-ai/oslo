from math import sqrt
from typing import Any, Dict

import torch.distributed as dist
import torch.nn as nn
from anytree import Node

from oslo.torch.distributed import ParallelMode
from oslo.torch.nn.parallel.pipeline_parallel._cost_estimator import (
    PartitioningCostEstimator,
)
from oslo.torch.nn.parallel.pipeline_parallel._utils import bfs, dfs


class ModelPartitioner(object):
    """
    Model Partitioner (Inter-module partitioning)

    Args:
        module (nn.Module): PyTorch module
        process_group (dist.ProcessGroup): process group object
        tracing_inputs (Dict[str, Any]): tracing input dictionary, will be input as **kwargs to model.
        memory_computation_balance (float): memory computation balance factor

    References:
        Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training
        https://arxiv.org/abs/2111.05972
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: dist.ProcessGroup,
        tracing_inputs: Dict[str, Any] = None,
        memory_computation_balance: float = 1.0,
    ):
        self.module = module
        self.process_group = process_group
        self.tracing_inputs = tracing_inputs
        self.memory_computation_balance = memory_computation_balance

        self.visited = {}
        self.bfs = {}
        self.partition_memoization = {}

    @staticmethod
    def _set_attribute(element, node):
        if hasattr(element, "oslo_parallel"):
            element.oslo_parallel[ParallelMode.PIPELINE] = node.device
        else:
            element.oslo_parallel = {ParallelMode.PIPELINE: node.device}

    def partition(self):
        # 1. construct tree
        self.root_node = Node(
            name=self.module.__class__.__qualname__,
            parent=None,
            modules=[self.module],
            parameters=self._get_parameters(self.module),
            execution_order=0,
            cost=1.0,
        )
        self._construct_tree(self.root_node, self.root_node.name)

        # 2. compute the partitioning cost
        cost_estimator = PartitioningCostEstimator(
            root_node=self.root_node,
            alpha=self.memory_computation_balance,
            tracing_inputs=self.tracing_inputs,
        )
        cost_estimator.compute_cost()

        # 3. partition tree
        self._tree_partitioning()

        # 4. set device to parameters and buffers
        for node in dfs(self.root_node):
            if all([not hasattr(child, "device") for child in node.children]):
                module = node.modules[0]
                self._set_attribute(module, node)
                for param in node.parameters:
                    self._set_attribute(param, node)
                for buffer in module.buffers():
                    self._set_attribute(buffer, node)

    @staticmethod
    def _get_parameters(module, to_list=True):
        parameters = module.parameters()
        parameters = list(parameters) if to_list else tuple(parameters)
        return parameters

    def _construct_tree(self, node, parent_name):
        for module in node.modules:
            for name, child in module.named_children():
                name = (
                    f"{parent_name}.{name}"
                    if parent_name != self.module.__class__.__qualname__
                    else name
                )
                parameters = self._get_parameters(child, to_list=False)

                if len(parameters) != 0:
                    visited_node = self.visited.get(parameters, None)
                    if visited_node and parent_name not in visited_node.name.split(","):
                        child_node = self.visited[parameters]
                        child_node.modules.append(child)
                        child_node.name = ",".join([child_node.name, name])
                        child_node.tied = True
                    else:
                        child_node = Node(
                            name=name,
                            parent=node,
                            modules=[child],
                            parameters=list(parameters),
                            tied=False,
                        )
                        self.visited[parameters] = child_node
                    self._construct_tree(child_node, name)

    @staticmethod
    def _partition_segments(sequence, k):
        n = len(sequence)
        if k >= n:
            return [[node] for node in sequence]
        _sum = [0] * (n + 1)
        dp = [[0 for _ in range(n + 1)] for _ in range(2)]
        tb = [[[] for _ in range(n + 1)] for _ in range(2)]

        for a in range(0, n):
            _sum[a + 1] = sequence[a].cost + _sum[a]
            dp[0][a + 1] = float("inf")
            dp[1][a + 1] = float("inf")

        for a in range(1, k + 1):
            for b in range(a + 1, n + 1):
                for c in range(a - 1, b):
                    if max(dp[(a - 1) % 2][c], _sum[b] - _sum[c]) < dp[a % 2][b]:
                        dp[a % 2][b] = max(dp[(a - 1) % 2][c], _sum[b] - _sum[c])
                        tb[a % 2][b] = tb[(a - 1) % 2][c] + [c]

        starts = tb[k % 2][n]
        ends = starts[1:] + [n]
        return [sequence[s:e] for s, e in zip(starts, ends)]

    # Algorithm 1
    def _tree_partitioning(self):
        # The algorithm starts with a set of virtual devices
        # P(r) = {0, 1, . . . , D − 1} for the root node r
        initial_partition = [p for p in range(self.process_group.size())]
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
            if n_children[k] < sqrt_mean_children:
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
