from typing import Union, Tuple, Optional
import math

from oslo.torch.distributed import ParallelMode

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F

from .lazy import LazyModuleMixin
from oslo.torch.utils import divide
from oslo.torch.distributed import (
    reduce_grad, 
    reduce_input,
    split_forward_gather_backward,
    gather_forward_split_backward,
)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool =False,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.skip_bias_add = skip_bias_add

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.skip_bias_add:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight), self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    

class LazyLinear(LazyModuleMixin, Linear):
    """
    Lazy initialized linear layer.

    This can be very helpful for model parallelism. When you initialize the model, If you use multiprocessing,
    multiple copies of paramters are copied to the CPU RAM, which causes the CPU RAM to run out.
    Therefore, after creating uninitialized parameters and re-adjusting them to a suitable size,
    you can initialize only the necessary parameters to a suitable GPU immediately.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.

    Notes:
        This is different from torch.nn.LazyLinear in terms of
        ``in_features`` can be input by user at the creation time.

    Examples:
        >>> from oslo.torch.nn import LazyLinear

        >>> layer = LazyLinear(2, 4)
        >>> print(layer.weight)
        <UninitializedParameter>

        >>> layer.initialize_parameters()
        >>> print(layer.weight)
        Parameter containing:
        tensor([[-0.7025,  0.5608],
                [-0.2529, -0.2636],
                [-0.5755, -0.2422],
                [ 0.4704,  0.6281]], requires_grad=True)
    """

    cls_to_become = Linear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        skip_bias_add: bool =False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(0, 0, False)
        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        self.weight = UninitializedParameter(**factory_kwargs)
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self) -> None:
        """Initialize parameters"""
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.cls_to_become is not None:
            self.__class__ = self.cls_to_become


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
        skip_bias_add: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        num_partition = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)
        
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        # Set up backprop all-reduce.
        input_parallel = reduce_grad(input_, ParallelMode.TENSOR_1D, self.parallel_context)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(output_parallel, ParallelMode.TENSOR_1D, dim=-1, parallel_context=self.parallel_context)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        else:
            return output
        

class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gather_output: bool = False,
        skip_bias_add: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        num_partition = self.parallel_context.get_world_size(ParallelMode.TENSOR_1D)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)
        
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        # Set up backprop all-reduce.
        if self.parallel_input:
            input_ = input_
        else:
            input_ = split_forward_gather_backward(input_, ParallelMode.PARALLEL_1D, dim=-1, parallel_context=self.parallel_context)
        output_parallel = F.linear(input_, self.weight)
        output = reduce_input(output_parallel, ParallelMode.TENSOR_1D, self.parallel_context)

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


class Linear2D(nn.Module):
    pass
