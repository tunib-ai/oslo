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
        "example": f"Manners maketh man. Do you {tokenizer.mask_token} what that means?",
        "output": lambda output: tokenizer.decode(output.logits.argmax(-1)[0]),
    },
    "sequence-classification": {
        "class": AutoModelForSequenceClassification.from_pretrained,
        "example": "I will decide how I feel, I will be happy today.",
        "output": lambda output: output.logits.argmax(-1).item(),
    },
    "causal-lm": {
        "class": AutoModelForCausalLM.from_pretrained,
        "example": "I don't want a lot for Christmas. There is just one thing",
        "output": lambda output: tokenizer.decode(output[0]),
    },
    "seq2seq-lm": {
        "class": AutoModelForSeq2SeqLM.from_pretrained,
        "example": "Life was like a box of chocolates. You never know what you’re gonna get.",
        "output": lambda output: tokenizer.decode(output[0]),
    },
}


assert args.task in TASKS, (
    f"{args.task} is not supported task. "
    f"Please choose one of {list(TASKS.keys())}. "
    "If there are no major problems, it will work for other tasks as well, "
    "but I haven't tested it, so if you encounter any problems, "
    "please report them through the github issue."
)

make_result = (
    lambda input, before, after: "\n"
    f"Result :\n"
    f"> Input: {input}\n"
    f"> Output (before OSLO): {TASKS[args.task]['output'](before)}\n"
    f"> Output (after OSLO): {TASKS[args.task]['output'](after)}\n"
)

# 3. Create a model and input
model = TASKS[args.task]["class"](args.model)
input = args.input if args.input is not None else TASKS[args.task]["example"]
forward_fn = model.forward if generation_task else partial(model.generate, num_beams=3)

# 4. Get result before parallelization
output_before = forward_fn(**tokenizer(input, return_tensors="pt"))

# 5. Parallelize the model
model = oslo.initialize(model, config=args.config)
forward_fn = model.forward if generation_task else partial(model.generate, num_beams=3)

# 6. Get result after parallelization
output_after = forward_fn(**tokenizer(input, return_tensors="pt").to("cuda"))

# 7. Print the results
print(make_result(input, output_before, output_after))
