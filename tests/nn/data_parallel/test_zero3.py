import wandb
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

from oslo.torch.distributed import ParallelContext
from oslo.torch.nn.parallel import FullyShardedDataParallel as FSDP
from oslo.torch.nn.parallel.utils import allocate_params
from oslo.torch.nn.parallel.distributed import (
    auto_wrap,
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=2,
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_zero = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).cuda()
model_zero = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).cuda()

with enable_wrap(wrapper_cls=FSDP, parallel_context=parallel_context):
    wrapper_fsdp = wrap(model_zero)

# allocate_params(wrapper_fsdp, parallel_context)
# allocate_params 함수는 추후에 모든 페러렐 래퍼를 관장하는 클래스에서 처리될 예정
# https://github.com/tunib-ai/oslo/blob/307131bbd5ed995ea8dca8ac541bfbce9bfec29b/oslo/pytorch/model_parallelism/model_parallel_engine.py

if dist.get_rank() == 0:
    print(wrapper_fsdp)

# 옵티마이저 생성
optimizer_no_zero = Adam(model_no_zero.parameters(), lr=3e-5)
optimizer_zero = Adam(wrapper_fsdp.parameters(), lr=3e-5)

# 데이터셋 생성
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=2)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name="zero3")

# 학습 시작
for data in dataloader:
    optimizer_zero.zero_grad()
    optimizer_no_zero.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_zero = wrapper_fsdp(**inputs, labels=inputs["input_ids"]).loss
    loss_no_zero = model_no_zero(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"ZERO3:{loss_zero}, NOTHING:{loss_no_zero}")
        wandb.log({"zero3": loss_zero, "nothing": loss_no_zero})

    loss_zero.backward()
    loss_no_zero.backward()

    optimizer_zero.step()
    optimizer_no_zero.step()
