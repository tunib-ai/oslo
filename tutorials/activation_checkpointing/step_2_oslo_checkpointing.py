"""
Activation checkpointing tutorial step 2:
How to use the oslo activation checkpointing for training?
"""
import torch
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import oslo

# 1. Initialize some variables
BATCH_SIZE = 128
SEQ_LEN = 128
TRAIN_STEP = 10

# 2. Create model and optimizer and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add pad token for batch training (GPT2 tokenizer doesn't have pad token)
tokenizer.pad_token = tokenizer.eos_token

# 3. Parallelize the model and turn on oslo activation checkpointing
model = oslo.initialize(
    model,
    config={
        "model_parallelism": {
            "enable": True,
            "tensor_parallel_size": 4,
        },
        "activation_checkpointing": {
            "enable": True,
            "cpu_checkpointing": True,
            "partitioned_checkpointing": True,
            "contiguous_checkpointing": True,
        },
    },
)

# 4. Load dataset and create data loader
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # 5. Make batch
    input_batch = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    ).to("cuda")

    # 6. Forward-Backward-Step
    loss = model(**input_batch, labels=input_batch["input_ids"], use_cache=False).loss
    if torch.distributed.get_rank() == 0:
        print(f"memory: {round(torch.cuda.memory_allocated() / (1024 ** 3), 4)}GiB")
        # memory: 6.681 GiB
    loss.backward()
    optimizer.step()

    if step > TRAIN_STEP:
        break
