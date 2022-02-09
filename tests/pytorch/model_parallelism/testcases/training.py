import os
import random
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.profiler import profile
from torch.profiler.profiler import ProfilerActivity
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import oslo


def tokenize(args, tokenizer, sample):
    return tokenizer(
        [str(sample)] * args.batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
    ).to("cuda")


def train_step(name, model, optimizer, inputs, step):
    output = model(**inputs)
    loss = output.loss
    loss.backward()
    optimizer.step()

    if dist.get_rank() == 0:
        wandb.log({name: loss}, step=step)


parser = ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--tokenizer", default=None, type=str)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--sequence_length", required=True, type=int)
parser.add_argument("--train_step", required=True, type=int)
parser.add_argument("--save_interval", required=True, type=int)
parser.add_argument("--tensor_parallel_size", default=1, type=int)
args = parser.parse_args()
args.tokenizer = args.tokenizer if args.tokenizer else args.model
name = (
    f"{args.model}-{args.task}-"
    f"tp={args.tensor_parallel_size}-"
    f"bsz={args.batch_size}-"
    f"len={args.sequence_length}"
)

# 1. Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Set random seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# 3. Define tasks and config
TASKS = {
    "sequence-classification": {
        "class": partial(
            AutoModelForSequenceClassification.from_pretrained, num_labels=3
        ),
        "load_dataset": lambda: load_dataset("multi_nli").data["train"],
        "preprocessing": lambda args, dataset: [
            (f"{str(p)}\n{str(h)}", l.as_py())
            for p, h, l in list(zip(dataset[2], dataset[5], dataset[9]))[
                : args.train_step
            ]
        ],
        "inputs": lambda args, tokenizer, sample: {
            "input_ids": tokenize(args, tokenizer, sample[0])["input_ids"],
            "labels": torch.tensor(sample[1])
            .unsqueeze(0)
            .repeat(args.batch_size, 1)
            .to("cuda"),
        },
    },
    "causal-lm": {
        "class": AutoModelForCausalLM.from_pretrained,
        "load_dataset": lambda: load_dataset("squad").data["train"]["context"],
        "preprocessing": lambda args, dataset: dataset[: args.train_step],
        "inputs": lambda args, tokenizer, sample: {
            "input_ids": tokenize(args, tokenizer, sample)["input_ids"],
            "labels": tokenize(args, tokenizer, sample)["input_ids"],
        },
    },
    "seq2seq-lm": {
        "class": AutoModelForSeq2SeqLM.from_pretrained,
        "load_dataset": lambda: load_dataset("wmt14", "de-en").data["train"][0],
        "preprocessing": lambda args, dataset: [
            (str(data[1]), str(data[0])) for data in dataset[: args.train_step]
        ],
        "inputs": lambda args, tokenizer, sample: {
            "input_ids": tokenize(args, tokenizer, sample[0])["input_ids"],
            "labels": tokenize(args, tokenizer, sample[1])["input_ids"],
        },
    },
}

OSLO_CONFIG = {
    "model_parallelism": {
        "enable": True,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
}


assert args.task in TASKS, (
    f"{args.task} is not supported task. "
    f"Please choose one of {list(TASKS.keys())}. "
    "If there are no major problems, it will work for other tasks as well, "
    "but I haven't tested it, so if you encounter any problems, "
    "please report them through the github issue."
)

# 4. Create parallelized model and optimizer
model_oslo = TASKS[args.task]["class"](args.model)
model_oslo = oslo.initialize(model_oslo, config=OSLO_CONFIG)
optimizer_oslo = Adam(params=model_oslo.parameters(), lr=3e-5)

# 5. Create non-parallelized model and optimizer
model_no_oslo = TASKS[args.task]["class"](args.model).to("cuda")
optimizer_no_oslo = Adam(params=model_no_oslo.parameters(), lr=3e-5)

# 6. Initialize wandb and create folders
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=name)
    os.makedirs("profile", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)

# 7. Load dataset and do preprocessing
dataset = TASKS[args.task]["load_dataset"]()
preprocessing = TASKS[args.task]["preprocessing"](args, dataset)

# 8. Start training
for i, sample in enumerate(tqdm(preprocessing)):
    if i > args.train_step:
        break

    # 9. Prepare input tokens
    inputs = TASKS[args.task]["inputs"](
        args,
        tokenizer,
        sample,
    )

    # 10. Do step for non-parallelized model
    train_step(
        name=f"no_oslo-{name}",
        model=model_no_oslo,
        optimizer=optimizer_no_oslo,
        inputs=inputs,
        step=i,
    )

    # 11. Do step for parallelized model
    train_step(
        name=f"oslo-{name}",
        model=model_oslo,
        optimizer=optimizer_oslo,
        inputs=inputs,
        step=i,
    )

    # 12. Save parallelized checkpoints
    if i % args.save_interval == 0:
        model_oslo.save_parallelized(
            save_directory="ckpt",
            save_config=True,
            merge_checkpoints=False,
        )
