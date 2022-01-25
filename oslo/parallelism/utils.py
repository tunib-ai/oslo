# Copyright 2021 TUNiB inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import psutil
import torch
from torch.fx import Node

NoneType = type(None)


def igetattr(obj, attr, *args):
    """
    Indexed getattr function

    Examples:
        >>> model = Model()
        >>> igetattr(model, "weight[2]")
    """
    if "[" in attr and "]" in attr:
        attr = "".join("\t".join(attr.split("[")).split("]")).split("\t")
        indexes = "[".join(attr[1:-1]).replace("[", "][")
        indexes = "[" + indexes + "]" if len(indexes) >= 1 else indexes
        return igetattr(obj, attr[0] + indexes)[int(attr[-1])]
    else:
        return getattr(obj, attr, *args)


def isetattr(obj, attr, val):
    """
    Indexed setattr function

    Examples:
        >>> model = Model()
        >>> isetattr(model, "weight[2]", new_weight)
    """
    if "[" in attr and "]" in attr:
        element = attr.split("[")[0]
        element_obj = getattr(obj, element)
        attr = "".join("\t".join(attr.split("[")).split("]")).split("\t")[1:]

        for i in range(len(attr) - 1):
            element_obj = element_obj[int(attr[i])]

        element_obj[int(attr[-1])] = val
    else:
        setattr(obj, attr, val)


def rgetattr(obj, attr, default=None):
    """
    Recursive getattr function based on igetattr

    Examples:
        >>> model = Model()
        >>> rgetattr(model, "layer[2].attention.weight[3].data")
    """

    try:
        left, right = attr.split(".", 1)
    except BaseException:
        return igetattr(obj, attr, default)
    return rgetattr(igetattr(obj, left), right, default)


def rsetattr(obj, attr, val):
    """
    Recursive setattr function based on isetattr

    Examples:
        >>> model = Model()
        >>> rgetattr(model, "layer[2].attention.weight[3].data", new_data)
    """

    try:
        left, right = attr.split(".", 1)
    except BaseException:
        return isetattr(obj, attr, val)
    return rsetattr(igetattr(obj, left), right, val)


def rhasattr(obj, attr):
    """
    Recursive hasattr function based on igetattr

    Examples:
        >>> model = Model()
        >>> rhasattr(model, "layer[2].attention.weight[3].data")
        True
    """

    try:
        left, right = attr.split(".", 1)
    except BaseException:
        return hasattr(obj, attr)
    try:
        get = igetattr(obj, left)
    except BaseException:
        return False
    return rhasattr(get, right)


def is_module(node: Node):
    return (
        node.op not in ["placeholder", "call_function", "call_method"]
        and "tensor_constant" not in node.target
    )


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def is_available_tracing(module):
    elem_size = torch.zeros(1, dtype=module.dtype).element_size()
    model_memory_size = num_params(module) * elem_size
    available_memory_size = psutil.virtual_memory().available
    return available_memory_size > model_memory_size * 2
    # multiply by 2 to consider act memory for safer tracing.
