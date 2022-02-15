import os
from argparse import ArgumentParser
from functools import partial

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import oslo

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--tokenizer", default=None, type=str)
parser.add_argument("--input", default=None, type=str)
parser.add_argument("--tensor_parallel_size", default=1, type=int)
args = parser.parse_args()
generation_task = args.task not in ["causal-lm", "seq2seq-lm"]
args.tokenizer = args.tokenizer if args.tokenizer else args.model

# 1. Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# 2. Define tasks and config
TASKS = {
    "masked-lm": {
        "class": AutoModelForMaskedLM.from_pretrained,
    },
    "sequence-classification": {
        "class": partial(
            AutoModelForSequenceClassification.from_pretrained, num_labels=3
        ),
    },
    "causal-lm": {
        "class": AutoModelForCausalLM.from_pretrained,
    },
    "seq2seq-lm": {
        "class": AutoModelForSeq2SeqLM.from_pretrained,
    },
}


assert args.task in TASKS, (
    f"{args.task} is not supported task. "
    f"Please choose one of {list(TASKS.keys())}. "
    "If there are no major problems, it will work for other tasks as well, "
    "but I haven't tested it, so if you encounter any problems, "
    "please report them through the github issue."
)

# 3. Create the model
model = TASKS[args.task]["class"](args.model)

# 4. Parallelize the model
model = oslo.initialize(model, config=args.config)

# 5. Load parallelized checkpoints
model = model.from_parallelized("ckpt")

# 6. Save parallelized checkpoints
model.save_parallelized("ckpt/merged", merge_checkpoints=True)
