import time
import wandb
import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.torch.nn.parallel.utils import allocate_params
import time


def time_trace(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end-start
    return wrapper


@time_trace
def fw(func, *args, **kwargs):
    return func(*args, **kwargs).loss


@time_trace
def bw(tensors):
    return tensors.backward()

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_1D,
)

model_name = "gpt2"
mkwargs = {
}
dataset_name = "squad"

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(model_name, **mkwargs)
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_tp = GPT2LMHeadModel(GPT2Config.from_pretrained(model_name)).cuda()
model_tp = GPT2LMHeadModel(GPT2Config.from_pretrained(model_name))
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
datasets = load_dataset(dataset_name).data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"{model_name}_tp1d_bs{batch_size}")
    cur = time.time()

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

    loss_no_tp, notp_fw_time = \
        fw(model_no_tp, **inputs, labels=inputs["input_ids"])
    loss_tp, tp_fw_time = \
        fw(wrapper_tp, **inputs, labels=inputs["input_ids"])

    fw_start_tp = time.time()
    loss_tp = wrapper_tp(**inputs, labels=inputs["input_ids"]).loss
    fw_time_tp = time.time() - fw_start_tp

    _, notp_bw_time = bw(loss_no_tp)
    _, tp_bw_time = bw(loss_tp)

    bw_start_tp = time.time()
    loss_tp.backward()
    optimizer_tp.step()
    bw_time_tp = time.time() - bw_start_tp

    if dist.get_rank() == 0:
        print(f"[tp/notp loss]: {loss_tp:.4f}, {loss_no_tp:.4f}")
        wandb.log(
            {
                "tp_loss": loss_tp,
                "notp_loss": loss_no_tp,
                "tp_fw_time": fw_time_tp,
                "notp_fw_time": notp_fw_time,
                "tp_bw_time": bw_time_tp,
                "notp_bw_time": notp_bw_time,
            }
        )

    if dist.get_rank() == 0:
        wandb.log({
            "tp.forward.time:": tp_fw_time,
            "tp.backward.time:": tp_bw_time,
            "notp.forward.time:": notp_fw_time,
            "notp.backward.time:": notp_bw_time})
    #
    # loss_tp = wrapper_tp(**inputs, labels=inputs["input_ids"]).loss
    # loss_no_tp = model_no_tp(**inputs, labels=inputs["input_ids"]).loss
    #
    # if dist.get_rank() == 0:
    #     print(f"TP:{loss_tp}, NOTP:{loss_no_tp}")
    #     wandb.log({"tp": loss_tp, "notp": loss_no_tp})
    #
    # loss_tp.backward()
    # loss_no_tp.backward()
    #
    # optimizer_tp.step()
    # optimizer_no_tp.step()

dist.barrier()
