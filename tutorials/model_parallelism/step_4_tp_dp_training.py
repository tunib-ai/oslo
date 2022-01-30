"""
Model parallelism tutorial step 4:
How to use the tensor + data parallelism for training?
"""
import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import oslo

# NOTE: This script must be executed with multiprocessing
# I strongly recommend to use torch.distributed.launch like this:
# ``python -m torch.distributed.launch --nproc_per_node 4 step_2_tp_training.py``

# 1. Initialize some variables
BATCH_SIZE = 4
SEQ_LEN = 64
SAVE_INTERVAL = 50
TRAIN_STEP = 100

# 2. Create model and optimizer and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add pad token for batch training (GPT2 tokenizer doesn't have pad token)
tokenizer.pad_token = tokenizer.eos_token

# 3. Parallelize the model
# Note that tp size  * dp size must be same or smaller than total num of gpus.
# If you have 4 GPUs, you can set tp size = 2 * dp size = 2.
# If you specify the tp size, the dp size will be determined automatically.

model = oslo.initialize(
    model, config={"model_parallelism": {"tensor_parallel_size": 2}}
)


# 4. Make the model data parallelizable
# We can use torch DDP module with OSLO TP.
# You can access various process groups and world sizes and ranks using ``model.mpu``.
# Refer to https://github.com/tunib-ai/oslo/blob/master/oslo/pytorch/model_parallelism/network/mpu.py
engine = DistributedDataParallel(
    module=model,
    process_group=model.mpu.get_data_parallel_group(),
    device_ids=[torch.cuda.current_device()],
    output_device=torch.cuda.current_device(),
)

# 5. Load dataset
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]

# 6. Create DistributedSampler to parallelize dataset
# You must specify the ``num_replicas`` and ``rank`` using ``model.mpu``.
sampler = DistributedSampler(
    dataset=datasets,
    shuffle=True,
    num_replicas=model.mpu.get_data_parallel_world_size(),
    rank=model.mpu.get_data_parallel_rank(),
)

# 7. Create data loader with sampler.
# Note you should turn off
dataloader = DataLoader(
    datasets,
    batch_size=BATCH_SIZE,
    shuffle=False,
    sampler=sampler,
)


for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # 8. Make batch
    input_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # 9. Forward-Backward-Step with DDP engine
    loss = engine(**input_batch, labels=input_batch["input_ids"]).loss
    if torch.distributed.get_rank() == 0:
        print(f"step:{step}, loss={loss}")
    loss.backward()
    optimizer.step()

    # 10. Save parallelized model
    # We support ``save_parallelized`` method.
    # This is similar with ``save_pretrained`` in the Transformers.
    # Checkpoints like 'pytorch_model_tp_0_pp_0.bin' will be saved.
    if step % SAVE_INTERVAL == 0:
        model.save_parallelized(save_directory="./parallel_ckpt")

    if step > TRAIN_STEP:
        break
