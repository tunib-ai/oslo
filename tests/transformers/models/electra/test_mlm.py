from oslo.transformers.tasks.data_masked_lm import (
    ProcessorForMaskedLM,
    DataCollatorForMaskedLM,
)
from oslo.transformers.models.electra.modeling_electra import ElectraForMaskedLM
from datasets import load_dataset
import os
import wandb
from torch.utils.data import DataLoader

try:
    from transformers.models.electra.modeling_electra import (
        ElectraForMaskedLM as TransformersElectraForMaskedLM,
    )
    from transformers import (
        Trainer,
        TrainingArguments,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def train(model, dataset, processor, data_collator, run_name, fp16):
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=40,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        seed=0,
        load_best_model_at_end=True,
        save_steps=400000,
        fp16=fp16,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=processor._tokenizer,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "test_models"

    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    model_name_or_path = "google/electra-small-generator"
    processor = ProcessorForMaskedLM(model_name_or_path)

    processed_dataset = dataset.map(
        processor, batched=True, remove_columns=dataset["train"].column_names
    )
    processed_dataset.cleanup_cache_files()

    data_collator = DataCollatorForMaskedLM(processor)

    oslo_model = ElectraForMaskedLM.from_pretrained(model_name_or_path)
    transformers_model = TransformersElectraForMaskedLM.from_pretrained(
        model_name_or_path
    )

    # Transformers
    train(
        model=transformers_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        run_name="Transformers Electra",
        fp16=False,
    )

    train(
        model=transformers_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        run_name="Transformers Electra FP16",
        fp16=True,
    )

    # # Oslo
    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        run_name="Oslo Electra",
        fp16=False,
    )

    train(
        model=oslo_model,
        dataset=processed_dataset,
        processor=processor,
        data_collator=data_collator,
        run_name="Oslo Electra FP16",
        fp16=True,
    )
