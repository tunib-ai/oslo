from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
    DataCollatorForCausalLM,
)
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
from oslo.transformers.models.bloom.modeling_bloom import (
    BloomForCausalLM,
    BloomForSequenceClassification,
)
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import os
import wandb

try:
    from transformers.models.bloom.modeling_bloom import (
        BloomForCausalLM as TransformersBloomForCausalLM,
        BloomForSequenceClassification as TransformersBloomForSequenceClassification,
    )
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoConfig,
    )
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


def train(model_name_or_path, run_name, fp16, task):
    if task == "clm":
        dataset, model, processor, data_collator = causal_lm(
            model_name_or_path, run_name
        )
        batch_size = 4
        eval_steps = 50
    elif task == "cls":
        dataset, model, processor, data_collator = sequence_classification(
            model_name_or_path, run_name
        )
        batch_size = 16
        eval_steps = 80
    else:
        raise ValueError("\ntask is one of ['clm', 'cls']\n")

    processed_dataset = dataset.map(
        processor, batched=True, remove_columns=dataset["train"].column_names
    )
    processed_dataset.cleanup_cache_files()

    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=0.1,
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
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
    )

    trainer.train()
    wandb.finish()


def sequence_classification(model_name_or_path, run_name):
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.rename_column("label", "labels")

    config = AutoConfig.from_pretrained(model_name_or_path)
    processor = ProcessorForSequenceClassification(model_name_or_path, 128)
    config.pad_token_id = processor._tokenizer.pad_token_id

    if "oslo" in run_name.lower():
        model = BloomForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
    else:
        model = TransformersBloomForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
    data_collator = DataCollatorForSequenceClassification(processor)

    return dataset, model, processor, data_collator


def causal_lm(model_name_or_path, run_name):
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.rename_column("sentence", "text")

    processor = ProcessorForCausalLM(model_name_or_path)
    data_collator = DataCollatorForCausalLM(processor)

    if "oslo" in run_name.lower():
        model = BloomForCausalLM.from_pretrained(model_name_or_path)
    else:
        model = TransformersBloomForCausalLM.from_pretrained(model_name_or_path)

    return dataset, model, processor, data_collator


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "test_models"
    model_name_or_path = "bigscience/bigscience-small-testing"

    for task in ["cls", "clm"]:
        # Transformers
        train(
            model_name_or_path,
            run_name="Transformers Bloom " + task,
            fp16=False,
            task=task,
        )

        train(
            model_name_or_path,
            run_name="Transformers Bloom FP16 " + task,
            fp16=True,
            task=task,
        )

        # Oslo
        train(
            model_name_or_path,
            run_name="Oslo Bloom " + task,
            fp16=False,
            task=task,
        )

        train(
            model_name_or_path,
            run_name="Oslo Bloom FP16 " + task,
            fp16=True,
            task=task,
        )
