"""
Model parallelism tutorial step 2:
How to use the tensor parallelism for training?
"""
import torch
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
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

# 2. Load dataset and create data loader
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(_) for _ in datasets[: TRAIN_STEP * BATCH_SIZE]]
dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

# 3. Create model and optimizer and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = Adam(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add pad token for batch training (GPT2 tokenizer doesn't have pad token)
tokenizer.pad_token = tokenizer.eos_token

# 4. Parallelize the model
# - ``tensor_parallel_size`` must be smaller then total num of gpus.
# - ``tensor_parallel_size`` must be power of 2. (e.g. 2, 4, 8, 16, ...)
# - ``tensor_parallel_size`` must be positive number.
model = oslo.initialize(
    model, config={"model_parallelism": {"tensor_parallel_size": 4}}
)

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
    loss = model(**input_batch, labels=input_batch["input_ids"]).loss
    if torch.distributed.get_rank() == 0:
        print(f"step:{step}, loss={loss}")
    loss.backward()
    optimizer.step()

    # 7. Save parallelized model
    # We support ``save_parallelized`` method.
    # This is similar with ``save_pretrained`` in the Transformers.
    # Checkpoints like 'pytorch_model_tp_0_pp_0.bin' will be saved.
    if step % SAVE_INTERVAL == 0:
        model.save_parallelized(save_directory="./parallel_ckpt")

    if step > TRAIN_STEP:
        break
