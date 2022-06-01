import inspect

import torch
from torch import nn
import torch.distributed as dist
from torch.cuda.amp import custom_fwd, custom_bwd

from oslo.torch.distributed.nn.functional import send, recv
from oslo.torch.distributed import ParallelMode
from oslo.torch.nn.parallel.utils import get_parallel_context

FORWARD_MAP = {}

### PP p2p com setup:
#   PPPreFwdP2PCom      PPPostFwdP2PCom
# -->  pre fwd --> fwd --> post fwd -->
# <-- post bwd <-- bwd <--  pre bwd <--

# TO DO:
#   torch.autograd.Function setup needs to optimized
#   Check how we need to set up the buffers
#   ?


### PP pre fwd p2p com


def _pp_pre_fwd_p2p_com(input_, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    print(f"pre_fwd, {input_.requires_grad}, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
    if input_.device.index != module_rank:  # device of input should be the same as module_rank
        if input_.device.index == rank: # if input_ is on our rank we send
            print(f"pre_fwd-before_send, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
            send(
              data=input_, 
              src_rank=rank, 
              dst_rank=module_rank, 
              parallel_context=parallel_context
            )
            print(f"pre_fwd-after_send, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        elif module_rank == rank: # if the module rank is our rank we receive input_
            print(f"pre_fwd-before_recv, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
            #input_ = torch.empty(input_.shape, device=rank) # TO DO: Check buffer preallocation!
            input_buffer = recv(
              src_rank=input_.device.index, 
              dst_rank=rank, 
              parallel_context=parallel_context
            )
            print(f"pre_fwd-after_recv, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        if input_.device.index == rank:
            return input_
        elif module_rank == rank: # if the module rank is our rank we receive input_
            return input_buffer
    else: # input_ and module are on the same rank
        print(f"pre_fwd-no_comm, {input_.requires_grad}, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        return input_


def _pp_post_bwd_p2p_com(input_grad, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    if input_grad.device.index != module_rank_parent:
        if module_rank == rank: # if input_grad is on our rank we send
            send(
              data=input_grad, 
              src_rank=rank, 
              dst_rank=module_rank_parent, 
              parallel_context=parallel_context
            )
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output
            input_grad = recv(
              src_rank=module_rank, 
              dst_rank=rank, 
              parallel_context=parallel_context
            )
        if module_rank_parent == rank: # if the parent rank is our rank we receive output
            return input_grad
    else:  # data_out and module are on the same rank
        return input_grad


class PPPreFwdP2PCom(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, module_rank, module_rank_parent, parallel_context, *args, **kwargs
    ):
        ctx.module_rank = module_rank
        ctx.module_rank_parent = module_rank_parent
        ctx.parallel_context = parallel_context

        output = []
        for x in args:
            if torch.is_tensor(x):
                y = _pp_pre_fwd_p2p_com(
                    x, module_rank, module_rank_parent, parallel_context
                )
                output.append(y)
            else:
                output.append(x)

        output_dict = dict()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                y = _pp_pre_fwd_p2p_com(
                    v, module_rank, module_rank_parent, parallel_context
                )
                output_dict[k] = y
            else:
                output_dict[k] = v

        return tuple(output), output_dict

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_output):
        module_rank = ctx.module_rank
        module_rank_parent = ctx.module_rank_parent
        parallel_context = ctx.parallel_context

        output = []
        for x in grad_output:
            y = _pp_post_bwd_p2p_com(
                x, module_rank, module_rank_parent, parallel_context
            )
            output.append(y)

        return (None, None, None) + tuple(output)


### PP post fwd p2p com


def _pp_post_fwd_p2p_com(output_, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    if output_.device.index != module_rank_parent:
        if module_rank == rank: # if output_ is on our rank we send
            send(
              data=output_, 
              src_rank=rank, 
              dst_rank=module_rank_parent, 
              parallel_context=parallel_context
            )
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output_
            output_ = recv(
              src_rank=module_rank, 
              dst_rank=rank, 
              parallel_context=parallel_context
            )
        if module_rank_parent == rank: # if the parent rank is our rank we receive output_
            return output_
    else:  # data_out and module are on the same rank
        print(f"_pp_post_fwd_p2p_com passed, {output_.requires_grad}, {id(output_)}")
        return output_


def _pp_pre_bwd_p2p_com(output_grad, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()

    print(f"backward!!! rank: {rank}, grad @ {output_grad.device}")

    if output_grad.device.index != module_rank:
        if output_grad.device.index == rank: # if output_grad is on our rank we send
            send(
              data=output_grad, 
              src_rank=rank, 
              dst_rank=module_rank, 
              parallel_context=parallel_context
            )
        elif module_rank == rank: # if the module rank is our rank we receive output_grad
            output_grad = recv(
              src_rank=output_grad.device.index, 
              dst_rank=rank, 
              parallel_context=parallel_context
            )
        if module_rank == rank: # if the module rank is our rank we receive output_grad
            return output_grad
    else:  # data_out_grad and module are on the same rank
        return output_grad


class PPPostFwdP2PCom(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, module_rank, module_rank_parent, parallel_context, *input_):
        ctx.module_rank = module_rank
        ctx.module_rank_parent = module_rank_parent
        ctx.parallel_context = parallel_context

        output = []
        for x in input_:
            if torch.is_tensor(x):
                y = _pp_post_fwd_p2p_com(
                    x, module_rank, module_rank_parent, parallel_context
                )
                output.append(y)
            else:
                output.append(x)

        return tuple(output)

    @staticmethod
    @custom_bwd
    def backward(ctx, *grad_output):
        module_rank = ctx.module_rank
        module_rank_parent = ctx.module_rank_parent
        parallel_context = ctx.parallel_context

        output = []
        for x in grad_output:
            y = _pp_pre_bwd_p2p_com(
                x, module_rank, module_rank_parent, parallel_context
            )
            output.append(y)

        return (None, None, None) + tuple(output)


def add_hook(module, parallel_context):
    rank = None
    rank_parent = None
    if hasattr(module, "oslo_parallel"):
        rank = module.oslo_parallel[ParallelMode.PIPELINE]
    if hasattr(module, "oslo_pp_parent_rank"):
        rank_parent = module.oslo_pp_parent_rank

    pre_com = None
    post_com = None
    if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
        assert rank is not None and rank_parent is not None
        pre_com = PPPreFwdP2PCom.apply
        post_com = PPPostFwdP2PCom.apply

    FORWARD_MAP[module.__class__.__qualname__ + str(id(module))] = (module.forward, rank, rank_parent, pre_com, post_com)

    def new_forward(*args, **kwargs):
        frame = inspect.currentframe()
        while hasattr(frame, "f_back"):
            f_locals = frame.f_locals
            if "self" in f_locals and isinstance(f_locals["self"], nn.Module):
                break
            else:
                frame = frame.f_back

        caller_module = frame.f_locals["self"]
        print(f'{caller_module.__class__.__qualname__} + {len(args)}')

        forward = FORWARD_MAP[caller_module.__class__.__qualname__ + str(id(caller_module))][0]
        rank = FORWARD_MAP[caller_module.__class__.__qualname__ + str(id(caller_module))][1]
        rank_parent = FORWARD_MAP[caller_module.__class__.__qualname__ + str(id(caller_module))][2]
        pre_com = FORWARD_MAP[caller_module.__class__.__qualname__ + str(id(caller_module))][3]
        post_com = FORWARD_MAP[caller_module.__class__.__qualname__ + str(id(caller_module))][4]

        if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
            assert pre_com is not None and post_com is not None
            args, kwargs = pre_com(rank, rank_parent, parallel_context, *args, **kwargs)
            x = forward(*args, **kwargs)

            # TODO; okay?
            #   What should we do with dict?
            wrapped = False
            if not isinstance(x, tuple):
                x = tuple([x])
                wrapped = True

            x = post_com(rank, rank_parent, parallel_context, *x)

            if wrapped:
                (x,) = x
        else:
            x = forward(*args, **kwargs)
        return x

    module.forward = new_forward
    setattr(module, "has_hook", True)


### nn.Module wrapper


class PPModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context=None)
        if hasattr(module, "oslo_parallel"):
            self.rank = module.oslo_parallel[ParallelMode.PIPELINE]
        # print(self.rank)
        # if module.node.name != "ROOT":
        if hasattr(module, "oslo_pp_parent_rank"):
            self.rank_parent = module.oslo_pp_parent_rank
        # else:
        #    self.rank_parent = None
        if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
            self.pre_com = PPPreFwdP2PCom(self.rank, self.rank_parent).apply
            self.post_com = PPPostFwdP2PCom(self.rank, self.rank_parent).apply
        # print(module)

    def __len__(self):
        return len(self.module)

    def forward(self, x, *args, **kwargs):
        # print(self.module)
        if hasattr(self.module, "oslo_parallel") and hasattr(
            self.module, "oslo_pp_parent_rank"
        ):
            print("fwd0", dist.get_rank(), self.rank, self.rank_parent, type(x))
            x = self.pre_com((x, self.rank, self.rank_parent, self.parallel_context))
            print("fwd1", dist.get_rank(), self.rank, self.rank_parent, type(x))
            x = self.module(x, *args, **kwargs)
            print("fwd2", dist.get_rank(), self.rank, self.rank_parent, type(x))
            x = self.post_com((x, self.rank, self.rank_parent, self.parallel_context))
            print("fwd3", dist.get_rank(), self.rank, self.rank_parent, type(x))
        else:
            x = self.module(x, *args, **kwargs)
        return x


def wrap_nn_modules(m, parallel_context):
    """Wraps every nn.Module object in a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not(hasattr(child, "has_hook")):
            add_hook(child, parallel_context)
        wrap_nn_modules(child, parallel_context)


def check_wrap_nn_modules(m):
    """Tests if every nn.Module object is wrapped within a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not (isinstance(child, PPModuleWrapper)):
            # assert that parent nn.Module is of type PPModuleWrapper
            assert isinstance(
                m, PPModuleWrapper
            ), "nn.Module object is not wrapped inside PPModuleWrapper object."
        check_wrap_nn_modules(child)
    return True
