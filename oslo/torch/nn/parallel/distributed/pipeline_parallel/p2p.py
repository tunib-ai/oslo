### PP p2p com setup:

# -->  pre fwd --> fwd --> post fwd -->
# <-- post bwd <-- bwd <--  pre bwd <--

# TO DO:
#   torch.autograd.Function setup needs to optimized
#   Check how we need to set up the buffers
#   rethink module wrapper setup (how to add/remove wrapper
#   ?


### PP pre fwd p2p com

def _pp_pre_fwd_p2p_com(input_, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if input_.device.index != module_rank:
        if input_.device.index == rank: # if input_ is on our rank we send
            request = dist.isend(tensor=data_in, dst=module_rank)
            request.wait()
        elif module_rank == rank: # if the module rank is our rank we receive input_
            input_buffer = torch.empty(input_.shape, device=rank)
            request = dist.irecv(tensor=input_buffer, src=input_.device.index)
            request.wait()
            return input_buffer
    else: # data_in and module are on the same rank
        return input_


def _pp_post_bwd_p2p_com(input_grad, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if input_grad.device.index != module_rank_parent:
        if module_rank == rank: # if input_grad is on our rank we send
            request = dist.isend(tensor=input_grad, dst=module_rank_parent)
            request.wait()
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output
            input_grad_buffer = torch.empty(input_grad.shape, device=rank)
            request = dist.irecv(tensor=input_grad_buffer, src=module_rank)
            request.wait()
            return input_grad_buffer
    else: # data_out and module are on the same rank
        return input_grad


class PPPreFwdP2PCom(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, module_rank=None, module_rank_parent=None):
        ctx.save_for_backward(module_rank, module_rank_parent)
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
            request.wait()
        elif module_rank_parent == rank: # if the parent rank is our rank we receive output_
            output_buffer = torch.empty(output_.shape, device=rank)
            request = dist.irecv(tensor=output_buffer, src=module_rank)
            request.wait()
            return output_buffer
    else: # data_out and module are on the same rank
        return output_


def _pp_pre_bwd_p2p_com(output_grad, module_rank, module_rank_parent):
    rank = dist.get_rank()
    if output_grad.device.index != module_rank:
        if data_out_grad.device.index == rank: # if output_grad is on our rank we send
            request = dist.isend(tensor=output_grad, dst=module_rank)
            request.wait()
        elif module_rank == rank: # if the module rank is our rank we receive output_grad
            output_grad_buffer = torch.empty(output_grad.shape, device=rank)
            request = dist.irecv(tensor=output_grad_buffer, src=output_grad.device.index)
            request.wait()
            return data_out_grad_buffer
    else: # data_out_grad and module are on the same rank
        return output_grad


class PPPostFwdP2PCom(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, module_rank=None, module_rank_parent=None):
        ctx.save_for_backward(module_rank, module_rank_parent)
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
        self.rank_parent = module.rank_parent
        self.pre_com = PPPreFwdP2PCom(self.rank, self.rank_parent).apply
        self.post_com = PPPostFwdP2PCom(self.rank, self.rank_parent).apply
        
    def forward(self, x):
        x = self.pre_com(x)
        x = self.module(x)
        x = self.post_com(x)
        return x
