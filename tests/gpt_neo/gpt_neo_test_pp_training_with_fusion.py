# Copyright 2021 TUNiB Inc.

import os
import random

import numpy
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from oslo.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
)


class TestPPTraining:
    def __init__(self, num_gpus, batch_size=1, total_step=3000):
        random.seed(42)
        numpy.random.seed(42)
        torch.manual_seed(42)
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.total_step = total_step
        self.save_path = "save"
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def get_grad(model):
        """For debugging"""
        param_per_layer = [
            (
                x[0].split(".mlp.c_fc.weight")[0].split("transformer.")[1],
                x[1].grad,
            )
            for x in model.named_parameters()
            if "mlp.c_fc.weight" in x[0]
        ]

        return param_per_layer

    @staticmethod
    def get_param(model):
        """For debugging"""
        param_per_layer = [
            (
                x[0].split(".mlp.c_fc.weight")[0].split("transformer.")[1],
                round(x[1].mean(-1)[0].item(), 5),
            )
            for x in model.named_parameters()
            if "mlp.c_fc.weight" in x[0]
        ]

        return param_per_layer

    @staticmethod
    def get_tied_param(model):
        """For debugging"""
        param_per_layer = [
            (x[0], round(x[1].mean(-1)[0].item(), 5))
            for x in model.named_parameters()
            if "wte" in x[0]
        ] + [
            (x[0], round(x[1].weight.data.mean(-1)[0].item(), 5))
            for x in model.named_children()
            if "lm_head" in x[0]
        ]
        return param_per_layer

    def test_gpt_neo_lm_head_model(self):
        datasets = load_dataset("squad").data["train"]["context"]
        datasets = [str(sample) for sample in datasets if len(str(sample)) < 500]
        data_loader = DataLoader(datasets, batch_size=self.batch_size, num_workers=8)

        model_pp = (
            GPTNeoForCausalLM.from_pretrained_with_parallel(
                "EleutherAI/gpt-neo-125M",
                pipeline_parallel_size=self.num_gpus,
            )
            .train()
            .fuse()
        )
        model_1d = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda()

        optimizer_1d = Adam(model_1d.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer_pp = Adam(model_pp.gpu_parameters(), lr=1e-4, weight_decay=1e-5)

        for i, data in enumerate(data_loader):
            optimizer_1d.zero_grad()
            optimizer_pp.zero_grad()

            tokens = self.tokenizer(
                data,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to("cuda")

            loss_pp = []
            for output in model_pp(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=tokens.input_ids,
                use_cache=False,
            ):
                micro_loss = output.loss
                micro_loss.backward()
                loss_pp.append(micro_loss.detach().item())

            loss_pp = sum(loss_pp) / len(loss_pp)

            loss_1d = model_1d(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=tokens.input_ids,
                use_cache=False,
            ).loss
            loss_1d.backward()

            if dist.get_rank() == 0:
                print(
                    f"GPTNeoForCausalLM - "
                    f"step: {i}, "
                    f"loss_1d:{loss_1d}, "
                    f"loss_pp:{loss_pp}"
                )

            optimizer_1d.step()
            optimizer_pp.step()

            if i >= self.total_step:
                break

            if i % 300 == 0:
                os.makedirs(self.save_path, exist_ok=True)
                model_pp.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/pp",
                    save_with_merging=False,
                )
                model_1d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/1d",
                    save_with_merging=False,
                )

    def test_gpt_neo_for_sequence_classification(self):
        datasets = load_dataset("multi_nli").data["train"]
        premise, hypothesis, label = datasets[2], datasets[5], datasets[9]
        datasets = [
            {"texts": str(p) + self.tokenizer.eos_token + str(h), "label": l.as_py()}
            for p, h, l in zip(premise, hypothesis, label)
        ]

        model_pp = (
            GPTNeoForSequenceClassification.from_pretrained_with_parallel(
                "EleutherAI/gpt-neo-125M",
                pipeline_parallel_size=self.num_gpus,
                num_labels=3,
            )
            .train()
            .fuse()
        )

        model_1d = (
            GPTNeoForSequenceClassification.from_pretrained(
                "EleutherAI/gpt-neo-125M", num_labels=3
            )
            .cuda()
            .train()
        )

        model_pp.config.pad_token_id = self.tokenizer.eos_token_id
        model_1d.config.pad_token_id = self.tokenizer.eos_token_id
        data_loader = DataLoader(datasets, batch_size=self.batch_size, num_workers=8)

        optimizer_1d = Adam(model_1d.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer_pp = Adam(model_pp.gpu_parameters(), lr=1e-3, weight_decay=1e-5)

        for i, data in enumerate(data_loader):
            optimizer_1d.zero_grad()
            optimizer_pp.zero_grad()

            tokens = self.tokenizer(
                data["texts"],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to("cuda")

            loss_pp = []
            for output in model_pp(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=data["label"].cuda(),
            ):
                micro_loss = output.loss
                micro_loss.backward()
                loss_pp.append(micro_loss.detach().item())

            loss_pp = sum(loss_pp) / len(loss_pp)

            loss_1d = model_1d(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=data["label"].cuda(),
            ).loss
            loss_1d.backward()

            if dist.get_rank() == 0:
                print(
                    f"GPTNeoForSequenceClassification - "
                    f"step: {i}, "
                    f"loss_1d:{loss_1d}, "
                    f"loss_pp:{loss_pp}"
                )

            optimizer_1d.step()
            optimizer_pp.step()

            if i >= self.total_step:
                break

            if i % 300 == 0:
                os.makedirs(self.save_path, exist_ok=True)
                model_pp.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/pp",
                    save_with_merging=False,
                )
                model_1d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/1d",
                    save_with_merging=False,
                )


if __name__ == "__main__":
    test = TestPPTraining(num_gpus=4, batch_size=8)
    test.test_gpt_neo_lm_head_model()
