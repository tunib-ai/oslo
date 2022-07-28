from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_t5_pretraining import (
    ProcessorForT5Pretraining,
    DataCollatorForT5Pretraining,
)
from oslo.transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from datasets import load_dataset
import torch
import os

try:
    from transformers.models.mt5.modeling_mt5 import (
        MT5ForConditionalGeneration as TransformersMT5ForConditionalGeneration,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def train(model, dataset, processor, data_collator, optimizers, run_name, fp16):
    args = OTrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        seed=0,
        load_best_model_at_end=True,
        save_steps=100000,
        fp16=fp16,
        run_name=run_name,
    )

    trainer = OTrainer(
        model=model,
        args=args,
        tokenizer=processor._tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "test_models"

    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    model_name_or_path = "t5-small"
    processor = ProcessorForT5Pretraining(model_name_or_path)

    processed_dataset = dataset.map(
        processor, batched=True, remove_columns=dataset["train"].column_names
    )
    processed_dataset.cleanup_cache_files()

    oslo_model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    transformers_model = TransformersMT5ForConditionalGeneration.from_pretrained(
        model_name_or_path
    )

    data_collator = DataCollatorForT5Pretraining(processor)

    oslo_optim = torch.optim.Adam(params=oslo_model.parameters(), lr=2e-5)
    oslo_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=oslo_optim, T_0=1
    )
    oslo_optimizers = (oslo_optim, oslo_scheduler)

    transformers_optim = torch.optim.Adam(
        params=transformers_model.parameters(), lr=2e-5
    )
    transformers_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=transformers_optim, T_0=1
    )
    transformers_optimizers = (transformers_optim, transformers_scheduler)

    # Transformers
    train(
        model=transformers_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=transformers_optimizers,
        run_name="Transformers MT5",
        fp16=False,
    )

    train(
        model=transformers_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=transformers_optimizers,
        run_name="Transformers MT5 FP16",
        fp16=True,
    )

    # Oslo
    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=oslo_optimizers,
        run_name="Oslo MT5",
        fp16=False,
    )

    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=oslo_optimizers,
        run_name="Oslo MT5 FP16",
        fp16=True,
    )
