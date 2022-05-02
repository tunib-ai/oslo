import torch
from torch import nn
import torch.distributed as dist

### PP p2p com setup:
#   PPPreFwdP2PCom      PPPostFwdP2PCom
# -->  pre fwd --> fwd --> post fwd -->
# <-- post bwd <-- bwd <--  pre bwd <--

# TO DO:
#   torch.autograd.Function setup needs to optimized
#   Check how we need to set up the buffers
#   ?


### PP pre fwd p2p com

def _pp_pre_fwd_p2p_com(input_, module_rank, module_rank_parent):
    rank = dist.get_rank()
    print(f"B, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
    if input_.device.index != module_rank:
        if input_.device.index == rank: # if input_ is on our rank we send
            print(f"C1a, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
            request = dist.isend(tensor=input_, dst=module_rank)
            print(f"C1b, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        elif module_rank == rank: # if the module rank is our rank we receive input_
            input_ = torch.empty(input_.shape, device=rank)
            request = dist.irecv(tensor=input_buffer, src=input_.device.index)
            print(f"C2, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        request.wait()
        return input_
    else: # input_ and module are on the same rank
        print(f"D, rank: {rank}, module_rank: {module_rank}, module_rank_parent: {module_rank_parent}, input_.device.index: {input_.device.index}")
        return input_


def _pp_post_bwd_p2p_com(input_grad, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if input_grad.device.index != module_rank_parent:
        if module_rank == rank: # if input_grad is on our rank we send
            request = dist.isend(tensor=input_grad, dst=module_rank_parent)
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output
            input_grad = torch.empty(input_grad.shape, device=rank)
            request = dist.irecv(tensor=input_grad_buffer, src=module_rank)
        request.wait()
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
        input_, module_rank, module_rank_parent = data
        print("PPPreFwdP2PCom", input_.device.index, module_rank, module_rank_parent)
        ctx.save_for_backward(
                torch.tensor(module_rank, device=module_rank),
                torch.tensor(module_rank_parent, device=module_rank),
                )
        print("A")
        return _pp_pre_fwd_p2p_com(input_, module_rank, module_rank_parent)

    @staticmethod
    def backward(ctx, grad_output):
        module_rank, module_rank_parent, = ctx.saved_tensors
        return _pp_post_bwd_p2p_com(grad_output, module_rank, module_rank_parent)
    
    
### PP post fwd p2p com

def _pp_post_fwd_p2p_com(output_, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if output_.device.index != module_rank_parent:
        if module_rank == rank: # if output_ is on our rank we send
            request = dist.isend(tensor=output_, dst=module_rank_parent)
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output_
            output_ = torch.empty(output_.shape, device=rank)
            request = dist.irecv(tensor=output_buffer, src=module_rank)
        request.wait()
        return output_
    else: # data_out and module are on the same rank
        return output_


def _pp_pre_bwd_p2p_com(output_grad, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if output_grad.device.index != module_rank:
        if data_out_grad.device.index == rank: # if output_grad is on our rank we send
            request = dist.isend(tensor=output_grad, dst=module_rank)
        elif module_rank == rank: # if the module rank is our rank we receive output_grad
            output_grad = torch.empty(output_grad.shape, device=rank)
            request = dist.irecv(tensor=output_grad_buffer, src=output_grad.device.index)
        request.wait()
        return data_out_grad
    else: # data_out_grad and module are on the same rank
        return output_grad


class PPPostFwdP2PCom(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, data):
        input_, module_rank, module_rank_parent = data
        ctx.save_for_backward(
                torch.tensor(module_rank, device=module_rank),
                torch.tensor(module_rank_parent, device=module_rank),
                )
        print("PPPostFwdP2PCom", input_.device.index, module_rank, module_rank_parent)
        return _pp_post_fwd_p2p_com(input_, module_rank, module_rank_parent)

    @staticmethod
    def backward(ctx, grad_output):
        module_rank, module_rank_parent, = ctx.saved_tensors
        return _pp_pre_bwd_p2p_com(grad_output, module_rank, module_rank_parent)


### nn.Module wrapper

class PPModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.rank = module.rank
        if hasattr(module, "rank_parent"):
            self.rank_parent = module.rank_parent
        else:
            self.rank_parent = None
        self.pre_com = PPPreFwdP2PCom(self.rank, self.rank_parent).apply
        self.post_com = PPPostFwdP2PCom(self.rank, self.rank_parent).apply
        
    def forward(self, x):
        print("fwd", self.rank, self.rank_parent)
        x = self.pre_com((x, self.rank, self.rank_parent))
        print(self.module)
        x = self.module(x)
        x = self.post_com((x, self.rank, self.rank_parent))
        return x


def wrap_nn_modules(m):
    """Wraps every nn.Module object in a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not(isinstance(child, PPModuleWrapper)):
            setattr(m, child_name, PPModuleWrapper(child))
        wrap_nn_modules(child)


def check_wrap_nn_modules(m):
    """Tests if every nn.Module object is wrapped within a PPModuleWrapper object."""
    for child_name, child in m.named_children():
        if isinstance(child, nn.Module) and not(isinstance(child, PPModuleWrapper)):
            # assert that parent nn.Module is of type PPModuleWrapper
            assert isinstance(m, PPModuleWrapper), "nn.Module object is not wrapped inside PPModuleWrapper object."
        check_wrap_nn_modules(child)
    return True

