from typing import Optional
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from src.processors import GPT2Processer
from src.utils import load_corpora

model_type_to_processor = {
    "gpt2": GPT2Processer,
}


@dataclass
class Arguments:
    model_type: str = field(
        default="gpt2",
        metadata={
            "choices": [
                "gpt2",
            ]
        },
    )
    tokenizer_dir: str = field(
        default="tokenizers/gpt2",
    )
    corpora_dir: str = field(
        default="corpora",
    )
    corpus_type: str = field(
        default="docu_json",
        metadata={
            "choices": [
                "docu_text",
                "docu_json",
                "sent_text",
                "sent_json",
            ]
        },
    )
    max_length: int = field(
        default=512,
    )
    batched: bool = field(
        default=False
    )
    num_proc: Optional[int] = field(
        default=None,
    )
    batch_size: int = field(
        default=1000,
    )
    writer_batch_size: int = field(
        default=1000,
    )
    load_from_cache_file: bool = field(
        default=True,
    )
    keep_in_memory: bool = field(
        default=False,
    )


def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    data_processor = model_type_to_processor[args.model_type](args.tokenizer_dir, args.max_length)

    corpora = load_corpora(args.corpora_dir, corpus_type=args.corpus_type)

    dataset = corpora.map(
        lambda examples: data_processor(examples["text"]),
        batched=args.batched,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        writer_batch_size=args.writer_batch_size,
        load_from_cache_file=args.load_from_cache_file,
        keep_in_memory=args.keep_in_memory,
        remove_columns=corpora.column_names,
    )

    dataset.save_to_disk(f"datasets/{args.model_type}")
    data_processor.save_tokenizer(f"datasets/{args.model_type}")


if __name__ == "__main__":
    main()