import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import Adam

from datasets import load_dataset
from torch.utils.data import DataLoader

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import PipelineParallel
from oslo.torch.nn.parallel.utils import allocate_params

parallel_context = ParallelContext.from_torch(pipeline_parallel_size=4)
#model = GPT2LMHeadModel.from_pretrained("gpt2")
model = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

wrapper_pp = PipelineParallel(
    model, parallel_context=parallel_context, memory_computation_balance=0.5, micro_batch_size=4
)

if dist.get_rank() == 0:
    print(wrapper_pp.partitioner.module)

optimizer_pp = Adam(wrapper_pp.parameters(), lr=3e-5)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=2)

allocate_params(wrapper_pp, parallel_context)
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

for data in dataloader:
    optimizer_pp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    #print(inputs["input_ids"])

    loss_pp = wrapper_pp(**inputs, labels=inputs["input_ids"]).loss
    #loss_no_pp = model(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"pp:{loss_pp}, NOTHING:{loss_pp}")
        #wandb.log({"pp": loss_zero, "nothing": loss_no_zero})

    loss_pp.backward()

    optimizer_pp.step()
