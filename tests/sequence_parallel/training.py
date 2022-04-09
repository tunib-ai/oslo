import os
import random
import torch
import torch.distributed as dist
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import wandb

from argparse import ArgumentParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sp_tokenize(args, tokenizer, sample):
    return tokenizer(
        [str(sample) * args.batch_size],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
    ).to("cuda")


def train_step(name, model, optimizer, inputs, step):
    output = model(**inputs)
    if isinstance(output, torch.Tensor):
        output.backward()
    else:
        loss = output.loss
        loss.backward()
    optimizer.step()

    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.log({name: loss}, step=step)


def train(
    args,
    dataset,
    # model_sp,
    # optimizer_sp,
    model_no_sp,
    optimizer_no_sp,
    tokenizer,
):
    for i, sample in enumerate(tqdm(dataset[: args.train_step])):
        if i > args.train_step:
            break

        inputs = {
            "input_ids": sp_tokenize(args, tokenizer, sample)["input_ids"],
            "labels": sp_tokenize(args, tokenizer, sample)["input_ids"],
        }

        train_step(
            name=f"no_sp_{args.model_name_or_path}",
            model=model_no_sp,
            optimizer=optimizer_no_sp,
            inputs=inputs,
            step=i,
        )

        # train_step(
        #     name=f"sp_{args.model_name_or_path}",
        #     model=model_sp,
        #     optimizer=optimizer_sp,
        #     inputs=inputs,
        #     step=i,
        # )

        # if i % args.save_interval == 0:
        #     if hasattr(model_sp, "save_parallelized"):
        #         model_sp.save_parallelized(
        #             save_directory="ckpt",
        #             save_config=True,
        #             merge_checkpoints=False,
        #         )


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dist.init_process_group(backend="nccl")

    parser = ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_name_or_path", required=True, type=str)  # "sshleifer/tiny-gpt2"
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--sequence_length", required=True, type=int)
    parser.add_argument("--train_step", required=True, type=int)
    parser.add_argument("--save_interval", required=True, type=int)
    args = parser.parse_args()

    seed_everything()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_no_sp = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cuda").train()
    optimizer_no_sp = Adam(params=model_no_sp.parameters(), lr=3e-5)

    # model_sp = GPT2WithSPLMHeadModel.from_pretrained(args.model_name_or_path)
    # optimizer_sp = Adam(params=model_sp.parameters(), lr=3e-5)

    project_name = f"{args.model_name_or_path}-bsz={args.batch_size}-len={args.sequence_length}"
    wandb.login()
    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb.init(
            project="oslo-sp",
            entity="sp",
            name=project_name,
        )
        os.makedirs("profile", exist_ok=True)
        os.makedirs("ckpt", exist_ok=True)

    dataset = load_dataset("squad").data["train"]["context"]

    train(
        args=args,
        dataset=dataset,
        # model_sp=model_sp,
        # optimizer_sp=optimizer_sp,
        model_no_sp=model_no_sp,
        optimizer_no_sp=optimizer_no_sp,
        tokenizer=tokenizer,
    )
