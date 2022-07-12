from oslo.transformers.training_args import TrainingArguments as OTrainingArguments
from oslo.transformers.trainer import Trainer as OTrainer
from oslo.transformers.tasks.data_roberta_pretraining import (
    ProcessorForRobertaPretraining,
    DataCollatorForRobertaPretraining,
)
from oslo.transformers.models.longformer.modeling_longformer import (
    LongformerForMaskedLM,
)
from datasets import load_dataset
import torch
import os

try:
    from transformers.models.longformer.modeling_longformer import (
        LongformerForMaskedLM as TransformersLongformerForMaskedLM,
    )
except ImportError:
    print("\nYou have to install `transformers` to use `oslo.transformers` modules\n")


def train(model, dataset, processor, data_collator, optimizers, run_name, fp16):
    args = OTrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        seed=0,
        load_best_model_at_end=True,
        save_steps=300000,
        fp16=fp16,
        run_name=run_name,
    )

    trainer = OTrainer(
        model=model,
        args=args,
        tokenizer=processor._tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
        train_dataset=dataset["train"].shard(7, 1),
        eval_dataset=dataset["validation"],
    )

    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "test_models"

    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    model_name_or_path = "allenai/longformer-base-4096"
    processor = ProcessorForRobertaPretraining(model_name_or_path)

    processed_dataset = dataset.map(
        processor, batched=True, remove_columns=dataset["train"].column_names
    )
    processed_dataset.cleanup_cache_files()

    oslo_model = LongformerForMaskedLM.from_pretrained(model_name_or_path)
    transformers_model = TransformersLongformerForMaskedLM.from_pretrained(
        model_name_or_path
    )

    data_collator = DataCollatorForRobertaPretraining(processor)

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
        run_name="Transformers Longformer",
        fp16=False,
    )

    train(
        model=transformers_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=transformers_optimizers,
        run_name="Transformers Longformer FP16",
        fp16=True,
    )

    # Oslo
    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=oslo_optimizers,
        run_name="Oslo Longformer",
        fp16=False,
    )

    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        optimizers=oslo_optimizers,
        run_name="Oslo Longformer FP16",
        fp16=True,
    )
