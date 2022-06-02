import torch.distributed as dist
import wandb
from datasets import load_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.utils import allocate_params
from _utils import split_batch_2d

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode=ParallelMode.TENSOR_2D,
)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# 모델 생성 및 병렬화 수행
model_no_tp = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2")).cuda()
model_tp = GPT2LMHeadModel(GPT2Config.from_pretrained("gpt2"))
wrapper_tp = TensorParallel(model_tp, parallel_context)
allocate_params(wrapper_tp, parallel_context)
# allocate_params 함수는 추후에 모든 페러렐 래퍼를 관장하는 클래스에서 처리될 예정
# https://github.com/tunib-ai/oslo/blob/307131bbd5ed995ea8dca8ac541bfbce9bfec29b/oslo/pytorch/model_parallelism/model_parallel_engine.py

if dist.get_rank() == 0:
    print(wrapper_tp)

# 옵티마이저 생성
optimizer_tp = Adam(wrapper_tp.parameters(), lr=3e-5)
optimizer_no_tp = Adam(model_no_tp.parameters(), lr=3e-5)

# 데이터셋 생성
batch_size = 16
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"tp2d_bs{batch_size}")

# 모니터링 생성 대기
dist.barrier()

# 학습 시작
for data in dataloader:
    optimizer_tp.zero_grad()
    optimizer_no_tp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_tp = wrapper_tp(**inputs, labels=inputs["input_ids"]).loss
    loss_no_tp = model_no_tp(**inputs, labels=inputs["input_ids"]).loss

    if dist.get_rank() == 0:
        print(f"TP:{loss_tp}, NOTP:{loss_no_tp}")
        wandb.log({"tp": loss_tp, "notp": loss_no_tp})

    loss_tp.backward()
    loss_no_tp.backward()

    optimizer_tp.step()
    optimizer_no_tp.step()

dist.barrier()
