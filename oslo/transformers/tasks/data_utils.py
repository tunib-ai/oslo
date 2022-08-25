import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from numpy.random import choice
from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
)
from oslo.transformers.tasks.data_masked_lm import (
    ProcessorForMaskedLM,
)
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
)
from oslo.transformers.tasks.data_token_classification import (
    ProcessorForTokenClassification,
)
from oslo.transformers.tasks.data_summarization import (
    ProcessorForSummarization,
)
from oslo.transformers.tasks.data_bert_pretraining import (
    ProcessorForBertPretraining,
)
from oslo.transformers.tasks.data_albert_pretraining import (
    ProcessorForAlbertPretraining,
)
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
)
from oslo.transformers.tasks.data_t5_pretraining import (
    ProcessorForT5Pretraining,
)

try:
    from transformers import AutoTokenizer
    from transformers.file_utils import ExplicitEnum
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


SENT_TEXT_SCRIPT = str(
    (Path(__file__).parent / "loading" / "sent_text.py").resolve().absolute()
)


class CorpusType(ExplicitEnum):
    DOCU_TEXT = "docu_text"
    DOCU_JSON = "docu_json"
    SENT_TEXT = "sent_text"
    SENT_JSON = "sent_json"
    DATASET = "dataset"


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = 1000,
):
    return (
        dataset[i : i + batch_size][key] for i in range(0, len(dataset), batch_size)
    )


def load_corpora(
    dir_path: str, corpus_type: str = "docu_json"
) -> Union[Dataset, DatasetDict]:
    corpora_dir = Path(dir_path).absolute()
    extension = corpus_type.split("_")[-1]

    if extension == "json":
        list_of_file_paths = [
            str(file_path) for file_path in corpora_dir.rglob("*.json")
        ]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.json")
    elif extension == "text":
        list_of_file_paths = [
            str(file_path) for file_path in corpora_dir.rglob("*.txt")
        ]
        if not list_of_file_paths:
            raise Exception("Check file extensions. Your files are not *.txt")
    else:
        raise Exception(f"{extension} is not supported.")

    if corpus_type == "docu_text":
        return load_dataset("text", data_files=list_of_file_paths, split="train")
    elif corpus_type == "docu_json":
        return load_dataset("json", data_files=list_of_file_paths, split="train")
    elif corpus_type == "sent_text":
        return load_dataset(
            SENT_TEXT_SCRIPT, data_files=list_of_file_paths, split="train"
        )
    elif corpus_type == "sent_json":
        raise NotImplementedError("sent_json will be supported soon.")
    else:
        raise ValueError(
            f"{corpus_type} must be one of ['docu_text', 'docu_json', 'sent_text', 'sent_json']"
        )


def train_tokenizer(
    model_name: str,
    vocab_size: int,
    min_frequency: int,
    corpora_dir: str,
    corpus_type: CorpusType,
    sampling_ratio: float,
    save_dir: Optional[str] = None,
    additional_special_tokens: Optional[List[str]] = None,
    batch_size: int = 1000,
):
    if save_dir is None:
        corpora_name = corpora_dir.split("/")[-1]
        save_dir = f"tokenizers/{model_name}_{corpora_name}"

    if corpus_type == "dataset":
        corpora = load_from_disk(corpora_dir)["train"]
    else:
        corpora = load_corpora(corpora_dir, corpus_type)

    assert sampling_ratio > 0, "Sampling_ratio must be greater than 0."

    if 0 < sampling_ratio < 1.0:
        total_size = len(corpora)
        sample_size = int(total_size * sampling_ratio)
        corpora = corpora.select(indices=choice(total_size, sample_size, replace=False))
    else:
        logging.warning("Since sampling_ratio >= 1.0, all corpora will be used.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_iterator = batch_iterator(corpora, batch_size=batch_size)

    if additional_special_tokens:
        assert len(additional_special_tokens) == len(
            set(additional_special_tokens)
        ), "Each additional special tokens must be unique."
        assert not set(tokenizer.all_special_tokens).intersection(
            set(additional_special_tokens)
        ), "Each additional special tokens are not of default special tokens from tokenizer."

        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            new_special_tokens=additional_special_tokens,
        )
    else:
        tokenizer = tokenizer.train_new_from_iterator(
            data_iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    tokenizer.save_pretrained(save_dir)


def serialize_corpora(
    model_type: str,
    task_type: str,
    tokenizer_dir: str,
    corpora_dir: str,
    corpus_type: CorpusType,
    max_length: int,
    mlm_probability: Optional[float] = 0.15,
    mean_noise_span_length: Optional[float] = 3.0,
    save_dir: Optional[str] = None,
    batched: bool = True,
    num_proc: Optional[int] = os.cpu_count(),
    batch_size: int = 1000,
    writer_batch_size: int = 1000,
    load_from_cache_file: bool = True,
    keep_in_memory: bool = False,
):
    if save_dir is None:
        save_dir = f"datasets/{model_type}_{task_type}"

    task_type_to_processor = {
        "causal_lm": ProcessorForCausalLM,
        "sequence_classification": ProcessorForSequenceClassification,
        "token_classification": ProcessorForTokenClassification,
        "summarization": ProcessorForSummarization,
        "bert_pretraining": ProcessorForBertPretraining,
        "albert_pretraining": ProcessorForAlbertPretraining,
        "roberta_pretraining": ProcessorForMaskedLM,
        "bart_pretraining": ProcessorForBartPretraining,
        "t5_pretraining": ProcessorForT5Pretraining,
    }

    if task_type not in task_type_to_processor:
        raise ValueError(f"{task_type} must be one of {task_type_to_processor.keys()}")

    if corpus_type == "dataset":
        corpora = load_from_disk(corpora_dir)["train"]
    else:
        corpora = load_corpora(corpora_dir, corpus_type)

    if "label" in corpora.column_names:
        corpora = corpora.rename_column("label", "labels")

    if "label_ids" in corpora.column_names:
        corpora = corpora.rename_column("label_ids", "labels")

    if task_type == "token_classification":
        data_processor = task_type_to_processor[task_type](
            tokenizer_dir, max_length, corpora
        )
    elif task_type == "t5_pretraining":
        data_processor = task_type_to_processor[task_type](
            tokenizer_dir,
            max_length,
            mlm_probability,
            mean_noise_span_length,
        )
    else:
        data_processor = task_type_to_processor[task_type](tokenizer_dir, max_length)

    dataset = corpora.map(
        data_processor,
        batched=batched,
        num_proc=num_proc,
        batch_size=batch_size,
        writer_batch_size=writer_batch_size,
        load_from_cache_file=load_from_cache_file,
        keep_in_memory=keep_in_memory,
        remove_columns=corpora.column_names,
    )

    dataset.save_to_disk(save_dir)
    data_processor.save_tokenizer(save_dir)
