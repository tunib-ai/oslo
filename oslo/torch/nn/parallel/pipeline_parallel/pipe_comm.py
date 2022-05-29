import torch
from torch import nn
import torch.distributed as dist
from oslo.torch.distributed.nn.functional import send, recv
from oslo.torch.distributed import ParallelMode
from oslo.torch.nn.parallel.utils import get_parallel_context

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
    print(f"B, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
    if input_.device.index != module_rank:  # device of input should be the same as module_rank
        if input_.device.index == rank: # if input_ is on our rank we send
            print(f"C1a, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
            send(data=input_, src_rank=module_rank, dst_rank=module_rank, parallel_context=parallel_context)
            print(f"C1b, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        elif module_rank == rank: # if the module rank is our rank we receive input_
            print(f"C2a, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
            #input_ = torch.empty(input_.shape, device=rank) # TO DO: Check buffer preallocation!
            input_buffer = recv(src_rank=input_.device.index, dst_rank=input_.device.index, parallel_context=parallel_context)
            print(f"C2b, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        if module_rank == rank: # if the module rank is our rank we receive input_
            return input_buffer
    else: # input_ and module are on the same rank
        print(f"D, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        return input_


def _pp_post_bwd_p2p_com(input_grad, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    if input_grad.device.index != module_rank_parent:
        if module_rank == rank: # if input_grad is on our rank we send
            send(data=input_grad, src_rank=module_rank_parent, dst_rank=module_rank_parent, parallel_context=parallel_context)
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output
            input_grad = recv(src_rank=module_rank, dst_rank=module_rank, parallel_context=parallel_context)
        if module_rank_parent == rank: # if the parent rank is our rank we receive output
            return input_grad
    else: # data_out and module are on the same rank
        return input_grad


class PPPreFwdP2PCom(torch.autograd.Function):
    def __init__(self, rank, rank_parent):
        super().__init__()
        self.rank = rank
        self.rank_parent = rank_parent

    @staticmethod
    def forward(ctx, data):
        input_, module_rank, module_rank_parent, parallel_context = data
        print("PPPreFwdP2PCom", input_.device.index, module_rank, module_rank_parent)
        ctx.save_for_backward(
                torch.tensor(module_rank, device=module_rank),
                torch.tensor(module_rank_parent, device=module_rank),
                parallel_context
                )
        print("A")
        return _pp_pre_fwd_p2p_com(input_, module_rank, module_rank_parent, parallel_context)
    #def forward(ctx, input_):
        #data, module_rank, module_rank_parent = input_
        #ctx.save_for_backward(module_rank, module_rank_parent)
        #return _pp_pre_fwd_p2p_com(input_, self.module_rank, self.module_rank_parent)

    @staticmethod
    def backward(ctx, grad_output):
        module_rank, module_rank_parent, parallel_context = ctx.saved_tensors
        return _pp_post_bwd_p2p_com(grad_output, module_rank, module_rank_parent, parallel_context)
    
    
### PP post fwd p2p com

def _pp_post_fwd_p2p_com(output_, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    if output_.device.index != module_rank_parent:
        if module_rank == rank: # if output_ is on our rank we send
            send(data=output_, src_rank=module_rank_parent, dst_rank=module_rank_parent, parallel_context=parallel_context)
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output_
            output_ = recv(src_rank=module_rank, dst_rank=module_rank, parallel_context=parallel_context)
        if module_rank_parent == rank: # if the parent rank is our rank we receive output_
            return output_
    else: # data_out and module are on the same rank
        return output_


def _pp_pre_bwd_p2p_com(output_grad, module_rank, module_rank_parent, parallel_context):
    rank = dist.get_rank()
    if output_grad.device.index != module_rank:
        if output_grad.device.index == rank: # if output_grad is on our rank we send
            send(data=output_grad, src_rank=module_rank, dst_rank=module_rank, parallel_context=parallel_context)
        elif module_rank == rank: # if the module rank is our rank we receive output_grad
            output_grad = recv(src_rank=output_grad.device.index, dst_rank=output_grad.device.index, parallel_context=parallel_context)
        if module_rank == rank: # if the module rank is our rank we receive output_grad
            return output_grad
    else: # data_out_grad and module are on the same rank
        return output_grad


class PPPostFwdP2PCom(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, data):
        input_, module_rank, module_rank_parent, parallel_context = data
        ctx.save_for_backward(
                torch.tensor(module_rank, device=module_rank),
                torch.tensor(module_rank_parent, device=module_rank),
                parallel_context
                )
        print("PPPostFwdP2PCom", input_.device.index, module_rank, module_rank_parent)
        return _pp_post_fwd_p2p_com(input_, module_rank, module_rank_parent, parallel_context)
    #def forward(ctx, input_):
        #data, module_rank, module_rank_parent = input_
        #ctx.save_for_backward(module_rank, module_rank_parent)
        #return _pp_post_fwd_p2p_com(input_, self.module_rank, self.module_rank_parent)

    @staticmethod
    def backward(ctx, grad_output):
        module_rank, module_rank_parent, parallel_context = ctx.saved_tensors
        return _pp_pre_bwd_p2p_com(grad_output, module_rank, module_rank_parent, parallel_context)


def add_hook(module):
    forward = module.forward
    parallel_context = get_parallel_context(module, parallel_context=None)

    rank = None
    rank_parent = None
    if hasattr(module, "oslo_parallel"):
        rank = module.oslo_parallel[ParallelMode.PIPELINE]
    if hasattr(module, "oslo_pp_parent_rank"):
        rank_parent = module.oslo_pp_parent_rank
    if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
        assert rank is not None and rank_parent is not None
        pre_com = PPPreFwdP2PCom(rank, rank_parent).apply
        post_com = PPPostFwdP2PCom(rank, rank_parent).apply

    def new_forward(*args, **kwargs):
        x = args[0]

        if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
            print("fwd0", dist.get_rank(), rank, rank_parent, type(x))

            x = pre_com((x, rank, rank_parent, parallel_context))
            print("fwd1", dist.get_rank(), rank, rank_parent, type(x))
            x = forward(x, *args[1:], **kwargs)
            print("fwd2", dist.get_rank(), rank, rank_parent, type(x))
            x = post_com((x, rank, rank_parent, parallel_context))
            print("fwd3", dist.get_rank(), rank, rank_parent, type(x))
        else:
            x = module(*args, **kwargs)
        return x

    module.forward = new_forward
    setattr(module, 'has_hook', True)

### nn.Module wrapper

class PPModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.parallel_context = get_parallel_context(module, parallel_context=None)
        if hasattr(module, "oslo_parallel"):
            self.rank = module.oslo_parallel[ParallelMode.PIPELINE]
        #print(self.rank)
        #if module.node.name != "ROOT":
        if hasattr(module, "oslo_pp_parent_rank"):
            self.rank_parent = module.oslo_pp_parent_rank
        #else:
        #    self.rank_parent = None
        if hasattr(module, "oslo_parallel") and hasattr(module, "oslo_pp_parent_rank"):
            self.pre_com = PPPreFwdP2PCom(self.rank, self.rank_parent).apply
            self.post_com = PPPostFwdP2PCom(self.rank, self.rank_parent).apply
        #print(module)
    
    def __len__(self):
        return len(self.module)

    def forward(self, x, *args, **kwargs):
        #print(self.module)
        if hasattr(self.module, "oslo_parallel") and hasattr(self.module, "oslo_pp_parent_rank"):
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


def wrap_nn_modules(m):
    """Wraps every nn.Module object in a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not(hasattr(child, "has_hook")):
            add_hook(child)
        wrap_nn_modules(child)


def check_wrap_nn_modules(m):
    """Tests if every nn.Module object is wrapped within a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not(isinstance(child, PPModuleWrapper)):
            # assert that parent nn.Module is of type PPModuleWrapper
            assert isinstance(m, PPModuleWrapper), "nn.Module object is not wrapped inside PPModuleWrapper object."
        check_wrap_nn_modules(child)
    return True
