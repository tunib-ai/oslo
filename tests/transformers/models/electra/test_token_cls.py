from oslo.transformers.tasks.data_token_classification import (
    ProcessorForTokenClassification,
    DataCollatorForTokenClassification,
)
from oslo.transformers.models.electra.modeling_electra import (
    ElectraForTokenClassification,
)
from datasets import load_dataset
import os
import wandb

try:
    from transformers.models.electra.modeling_electra import (
        ElectraForTokenClassification as TransformersElectraForTokenClassification,
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
        eval_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        seed=0,
        load_best_model_at_end=True,
        save_steps=100000,
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

    dataset = load_dataset("conll2003")
    dataset = dataset.rename_column("ner_tags", "labels")

    model_name_or_path = "google/electra-small-discriminator"
    processor = ProcessorForTokenClassification(model_name_or_path, 128, dataset)

    processed_dataset = dataset.map(
        processor, batched=True, remove_columns=dataset["train"].column_names
    )
    processed_dataset.cleanup_cache_files()

    oslo_model = ElectraForTokenClassification.from_pretrained(
        model_name_or_path,
        id2label=processor.id2label,
        label2id=processor.label2id,
    )
    transformers_model = TransformersElectraForTokenClassification.from_pretrained(
        model_name_or_path,
        id2label=processor.id2label,
        label2id=processor.label2id,
    )

    data_collator = DataCollatorForTokenClassification(processor)

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

    # Oslo
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
