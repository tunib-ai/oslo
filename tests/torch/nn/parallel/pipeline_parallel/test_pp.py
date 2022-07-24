import concurrent.futures
from threading import Thread

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params


# _print = print
#
#
# def print(*args, **kw):
#     if dist.get_rank() == 0:
#         _print(*args, **kw)


torch.manual_seed(42)

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
)

current_device = torch.cuda.current_device()


# def print_device(i):
#     torch.cuda.set_device(torch.distributed.get_rank())
#     print(f'{torch.cuda.current_device()}')
#     dummy = torch.rand(1)
#
#     print(dummy)
#     dummy = dummy.cuda()
#
#     print(dummy.device)
#
#
# with concurrent.futures.ThreadPoolExecutor() as exe:
#     for i in range(1):
#         exe.submit(print_device, i)

# t = Thread(
#     target=print_device,
#     args=(0, )
# )
# t.start()

n_steps = 1
batch_size = 16

in_channels = 16
hidden_channels = 8
out_channels = 4


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)

        self.fcs = nn.ModuleList()
        for _ in range(60):
            self.fcs.append(
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)

        for layer in self.fcs:
            x = layer(x)

        x = self.fc2(x)
        return x

# fc1 = nn.Linear(in_channels, hidden_channels)
# fc2 = nn.Linear(hidden_channels, out_channels)
# model = nn.Sequential(fc1, fc2)

# fc1_no_pp = nn.Linear(in_channels, hidden_channels)
# fc2_no_pp = nn.Linear(hidden_channels, out_channels)
# model_no_pp = nn.Sequential(fc1_no_pp, fc2_no_pp)
# model_no_pp.load_state_dict(model.state_dict())
# model_no_pp.to(current_device)

model = SmallModel()
model_no_pp = SmallModel()
model_no_pp.load_state_dict(model.state_dict())
model_no_pp.to(current_device)

# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model_no_pp = GPT2LMHeadModel.from_pretrained("gpt2")
# model_no_pp.load_state_dict(model.state_dict())
# model_no_pp.to(current_device)

wrapper_pp = PipelineParallel(
    model,
    parallel_context=parallel_context,
    memory_computation_balance=1.0,
)

wrapper_pp.train()

if parallel_context.get_global_rank() == 0:
    print(wrapper_pp.partitioner.module)

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-5)
optimizer_no_pp = Adam(model_no_pp.parameters(), lr=3e-5)


from oslo.torch.nn.parallel.pipeline_parallel._hooks import wrap_forward

from torch.nn.modules.batchnorm import SyncBatchNorm


allocate_params(wrapper_pp, parallel_context)


def print_location(module, prefix):
    for n, m in module.named_children():
        new_prefix = f'{prefix}.{n}' if prefix != '' else n
        print(new_prefix, m.oslo_parallel[ParallelMode.PIPELINE])
        print_location(m, new_prefix)


print_location(wrapper_pp, '')


print(f"FC1 @ {model.fc1.weight.device}, FCS[-1] @ {model.fcs[-1].weight.device}")
print(f"FC1 @ {model.fc1.location}, FCS[-1] @ {model.fcs[-1].location}")
print(f"FC1 @ {wrapper_pp.module.fc1.weight.device}, FCS[-1] @ {wrapper_pp.module.fcs[-1].weight.device}")



def hook_fn(m, i):
    print(f'{m.weight.device}, {m.location} \n')


def get_all_layers(net):
    for name, layer in net.named_modules():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        elif isinstance(layer, nn.Linear):
            # it's a non sequential. Register a hook
            layer.register_forward_pre_hook(hook_fn)


get_all_layers(wrapper_pp)


"""
for rank in range(dist.get_world_size()):
    if dist.get_rank() == rank:
        print(f"RANK: {rank}:")
        num_params = 0
        for name, param in wrapper_pp.named_parameters():
            if param.device != torch.device("cpu"):
                print(f"> {name}: {param.device}")
                num_params += param.numel()
            else:
                print(f"> {name}: {param.device}")
        print(f"RANK {rank} params: {num_params}")
    dist.barrier()
    print()
"""


@record
def run():
    def recursive_convert(module):
        wrap_forward(module)

        for n, m in module.named_children():
            recursive_convert(m)

    recursive_convert(wrapper_pp.module)  # CAUTION; PipelineParallel's forward need to be remained!

    loss_fn = torch.nn.MSELoss()


    for i in range(n_steps):
        sample_input = torch.rand(batch_size, in_channels).cuda()
        sample_output = torch.rand(batch_size, out_channels)

        optimizer_pp.zero_grad()
        optimizer_no_pp.zero_grad()

        out_pp = wrapper_pp(sample_input)
        out_no_pp = model_no_pp(sample_input)

        print(out_pp)

run()
#
#     if out_pp is not None:
#         sample_output = sample_output.to(out_pp.device)
#         loss_pp = loss_fn(out_pp, sample_output)
#         loss_no_pp = loss_fn(out_no_pp, sample_output)
#
#         # if parallel_context.get_global_rank() == 0:
#         _print(f"rank: {dist.get_rank()}, pp:{loss_pp}, NOTHING:{loss_no_pp}")
#     _print("AHHHHHHHHHHH!!!!!!!!!!!")
#
#     dist.barrier()
#
#     _print(f"rank: {dist.get_rank()}, barrier")
#
#     if out_pp is not None:
#         loss_pp.backward()
#         loss_no_pp.backward()
#
#     if out_pp is not None:
#         for n, p in wrapper_pp.named_parameters():
#             print(n, p.grad)
#
#     optimizer_pp.step()
#     optimizer_no_pp.step()
