from typing import List, Tuple

import torch
import torch.nn as nn


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


def get_parameter_dtype(parameter: nn.Module):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def is_huggingface_model(model):
    try:
        import transformers

        return isinstance(model, transformers.PreTrainedModel)
    except ImportError:
        return False
