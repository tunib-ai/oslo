# Copyright 2021 TUNiB Inc.

import os
import random

import numpy
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer

from oslo.models.gptj.modeling_gptj import (
    GPTJForCausalLM,
    GPTJForSequenceClassification,
)


class TestTPTrainingWithDDP:
    def __init__(self, num_gpus, batch_size=4, total_step=3000):
        random.seed(42)
        numpy.random.seed(42)
        torch.manual_seed(42)
        self.save_path = "save"
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.total_step = total_step
        self.tokenizer = GPT2Tokenizer.from_pretrained("anton-l/gpt-j-tiny-random")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def get_grad(model):
        """For debugging"""
        param_per_layer = [
            (
                x[0].split(".mlp.fc_in.weight")[0].split("transformer.")[1],
                x[1].grad,
            )
            for x in model.named_parameters()
            if "mlp.fc_in.weight" in x[0]
        ]

        return param_per_layer

    @staticmethod
    def get_param(model):
        """For debugging"""
        param_per_layer = [
            (
                x[0].split(".mlp.fc_in.weight")[0].split("transformer.")[1],
                round(x[1].mean(-1)[0].item(), 5),
            )
            for x in model.named_parameters()
            if "mlp.fc_in.weight" in x[0]
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

    def test_gptj_lm_head_model(self):
        model_tp = GPTJForCausalLM.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random", tensor_parallel_size=self.num_gpus // 2
        )
        model_1d = GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random").cuda()

        optimizer_1d = Adam(model_1d.parameters(), lr=1e-5, weight_decay=1e-5)
        optimizer_tp = Adam(model_tp.gpu_parameters(), lr=1e-5, weight_decay=1e-5)

        data_parallel_group = model_tp.mpu.get_data_parallel_group()
        data_parallel_size = model_tp.mpu.get_data_parallel_world_size()
        data_parallel_rank = model_tp.mpu.get_data_parallel_rank()

        curr_device = torch.cuda.current_device()
        model_1d_ddp = DistributedDataParallel(
            model_1d,
            process_group=data_parallel_group,
            device_ids=[curr_device],
            output_device=curr_device,
        )

        model_tp_ddp = DistributedDataParallel(
            model_tp.gpu_modules(),
            process_group=data_parallel_group,
            device_ids=[curr_device],
            output_device=curr_device,
        )

        datasets = load_dataset("squad").data["train"]["context"]
        datasets = [str(sample) for sample in datasets if len(str(sample)) < 500]

        data_loader = DataLoader(
            datasets,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(
                datasets,
                num_replicas=data_parallel_size,
                rank=data_parallel_rank,
                shuffle=False,
            ),
        )

        for i, data in enumerate(data_loader):
            optimizer_1d.zero_grad()
            optimizer_tp.zero_grad()

            tokens = self.tokenizer(
                data,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )

            loss_tp = model_tp_ddp(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=tokens.input_ids.cuda(),
            ).loss

            loss_1d = model_1d_ddp(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=tokens.input_ids.cuda(),
            ).loss

            if dist.get_rank() == 0:
                print(
                    f"GPTJForCausalLM - "
                    f"step: {i}, "
                    f"loss_1d:{loss_1d}, "
                    f"loss_tp:{loss_tp}"
                )

            loss_1d.backward()
            loss_tp.backward()
            optimizer_1d.step()
            optimizer_tp.step()

            if i >= self.total_step:
                break

            if i % 300 == 0:
                os.makedirs(self.save_path, exist_ok=True)
                model_tp.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/tp",
                    save_with_merging=False,
                )
                model_1d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/1d",
                    save_with_merging=False,
                )

    def test_gptj_for_sequence_classification(self):
        model_tp = GPTJForSequenceClassification.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            tensor_parallel_size=self.num_gpus,
            num_labels=3,
        )
        model_1d = GPTJForSequenceClassification.from_pretrained(
            "anton-l/gpt-j-tiny-random", num_labels=3
        ).cuda()

        model_1d.config.pad_token_id = self.tokenizer.eos_token_id
        model_tp.config.pad_token_id = self.tokenizer.eos_token_id

        optimizer_1d = Adam(model_1d.parameters(), lr=1e-5, weight_decay=1e-5)
        optimizer_tp = Adam(model_tp.gpu_parameters(), lr=1e-5, weight_decay=1e-5)

        data_parallel_group = model_tp.mpu.get_data_parallel_group()
        data_parallel_size = model_tp.mpu.get_data_parallel_world_size()
        data_parallel_rank = model_tp.mpu.get_data_parallel_rank()
        curr_device = torch.cuda.current_device()

        model_1d_ddp = DistributedDataParallel(
            model_1d,
            process_group=data_parallel_group,
            device_ids=[curr_device],
        )

        model_tp_ddp = DistributedDataParallel(
            model_tp.gpu_modules(),
            process_group=data_parallel_group,
            device_ids=[curr_device],
        )

        datasets = load_dataset("multi_nli").data["train"]
        premise, hypothesis, label = datasets[2], datasets[5], datasets[9]
        datasets = [
            {"texts": str(p) + self.tokenizer.eos_token + str(h), "label": l.as_py()}
            for p, h, l in zip(premise, hypothesis, label)
        ]

        data_loader = DataLoader(
            datasets,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(
                datasets,
                num_replicas=data_parallel_size,
                rank=data_parallel_rank,
                shuffle=False,
            ),
        )

        for i, data in enumerate(data_loader):
            optimizer_1d.zero_grad()
            optimizer_tp.zero_grad()

            tokens = self.tokenizer(
                data["texts"],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )

            loss_tp = model_tp_ddp(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=data["label"].cuda(),
            ).loss

            loss_1d = model_1d_ddp(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=data["label"].cuda(),
            ).loss

            if dist.get_rank() == 0:
                print(
                    f"GPTJForSequenceClassification - "
                    f"step: {i}, "
                    f"loss_1d:{loss_1d}, "
                    f"loss_tp:{loss_tp}"
                )

            loss_1d.backward()
            loss_tp.backward()
            optimizer_1d.step()
            optimizer_tp.step()

            if i >= self.total_step:
                break

            if i % 300 == 0:
                os.makedirs(self.save_path, exist_ok=True)
                model_tp.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/tp",
                    save_with_merging=False,
                )
                model_1d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/1d",
                    save_with_merging=False,
                )


if __name__ == "__main__":
    test = TestTPTrainingWithDDP(num_gpus=4)
    test.test_gptj_lm_head_model()
    # num_gpus=4 ==> tp=2 * dp=2
