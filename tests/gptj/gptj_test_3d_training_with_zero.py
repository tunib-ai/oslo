# Copyright 2021 TUNiB Inc.

import os
import random

import numpy
import torch
import torch.distributed as dist
from datasets import load_dataset
from deepspeed import deepspeed
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer

from oslo.models.gptj.modeling_gptj import GPTJForCausalLM


class Test3DTrainingWithZeRO:
    def __init__(self, num_gpus, batch_size=1, total_step=3000):
        random.seed(42)
        numpy.random.seed(42)
        torch.manual_seed(42)
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.total_step = total_step
        self.save_path = "save"
        self.tokenizer = GPT2Tokenizer.from_pretrained("anton-l/gpt-j-tiny-random")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ds_config = {
            "train_batch_size": self.batch_size,
            "fp16": {"enabled": True},
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.total_step,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 3e-5,
                    "warmup_num_steps": self.total_step * 0.1,
                },
            },
            "zero_optimization": {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 2,
                "synchronize_checkpoint_boundary": True,
                "profile": False,
            },
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,
            "step_per_print": 99999999,
        }

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
        model_3d = GPTJForCausalLM.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random",
            pipeline_parallel_size=self.num_gpus // 4,
            tensor_parallel_size=self.num_gpus // 4,
        )
        model_1d = GPTJForCausalLM.from_pretrained_with_parallel(
            "anton-l/gpt-j-tiny-random"
        )

        data_parallel_size = model_3d.mpu.get_data_parallel_world_size()
        data_parallel_rank = model_3d.mpu.get_data_parallel_rank()

        optimizer_1d = Adam(model_1d.parameters(), lr=1e-5, weight_decay=1e-5)
        optimizer_3d = Adam(model_3d.gpu_parameters(), lr=1e-5, weight_decay=1e-5)

        engine_1d, optimizer_1d, _, scheduler_1d = deepspeed.initialize(
            model=model_1d,
            optimizer=optimizer_1d,
            mpu=model_3d.mpu,
            config=self.ds_config,
        )
        engine_3d, optimizer_3d, _, scheduler_3d = deepspeed.initialize(
            model=model_3d.gpu_modules(),
            optimizer=optimizer_3d,
            mpu=model_3d.mpu,
            config=self.ds_config,
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
            optimizer_3d.zero_grad()

            tokens = self.tokenizer(
                data,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )

            loss_3d = []
            for output in engine_3d(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=tokens.input_ids.cuda(),
            ):
                loss = output.loss
                engine_3d.backward(loss)
                loss_3d.append(loss.detach().item())

            loss_3d = sum(loss_3d) / len(loss_3d)

            loss_1d = engine_1d(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda(),
                labels=tokens.input_ids.cuda(),
            ).loss
            engine_1d.backward(loss_1d)

            engine_3d.step()
            engine_1d.step()

            if dist.get_rank() == 0:
                print(
                    f"GPTJForCausalLM - "
                    f"step: {i}, loss_1d:{loss_1d}, loss_3d:{loss_3d}"
                )

            if i >= self.total_step:
                break

            if i % 300 == 0:
                os.makedirs(self.save_path, exist_ok=True)
                model_1d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/1d",
                    save_with_merging=False,
                )
                model_3d.save_pretrained_with_parallel(
                    save_directory=self.save_path + "/3d",
                    save_with_merging=False,
                )


if __name__ == "__main__":
    test = Test3DTrainingWithZeRO(num_gpus=8, batch_size=16)
    test.test_gptj_lm_head_model()
    # num_gpus=8 ==> pp=2 * dp=2 * tp=2
