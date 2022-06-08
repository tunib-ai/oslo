import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params


_print = print

def print(*args, **kw):
    if dist.get_rank() == 0:
        _print(*args, **kw)


torch.manual_seed(42)

parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
)

current_device = torch.cuda.current_device()


config = GPT2Config.from_pretrained("gpt2")
config.use_cache = False
model = GPT2LMHeadModel(config)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=2)

n_steps = 20
batch_size = 16

wrapper_pp = PipelineParallel(
    model,
    parallel_context=parallel_context,
    micro_batch_size=4,
    use_auto_partitioning=True,
    memory_computation_balance=1.0,
)

wrapper_pp.train()

if parallel_context.get_global_rank() == 0:
    print(wrapper_pp.partitioner.module)

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-5)

allocate_params(wrapper_pp, parallel_context)


def hook_fn(m, i, o):
    print(m)


def get_all_layers(net):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            # it's a non sequential. Register a hook
            layer.register_backward_hook(hook_fn)


get_all_layers(wrapper_pp)

loss_fn = torch.nn.MSELoss()

for data in dataloader:
    optimizer_pp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_pp = wrapper_pp(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"pp:{loss_pp}")

    loss_pp.backward()

    optimizer_pp.step()
