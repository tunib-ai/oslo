import os
import argparse
from transformers import AutoTokenizer
from oslo import DatasetPreprocessor


def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True)
parser.add_argument('--small_first', action='store_true', default=False)
parser.add_argument('--end_file_record_path', default='end_files.txt')
parser.add_argument('--end_dir_record_path', default='end_dirs.txt')
args = parser.parse_args()

dir_sizes = list()
dirs = os.listdir(args.base_dir)

for dir_ in dirs:
    dir_sizes.append(get_dir_size(os.path.join(args.base_dir, dir_)))

if args.small_first:
    zipped = zip(dir_sizes, dirs)
    zipped = sorted(zipped)
    sorted_dir_sizes, sorted_dirs = zip(*zipped)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained('../gpt2')

preprocessor = DatasetPreprocessor(
    tokenizer=tokenizer,
    binarization_impl="mmap",
    eod_token_id=tokenizer.eos_token_id,
    append_eod=True,
)

with open(args.end_file_record_path, 'w') as f, open(args.end_dir_record_path, 'w') as ff:
    for dir_ in dirs:
        for file in os.listdir(os.path.join(args.base_dir, dir_)):
            if file.endswith('jsonl'):
                path = os.path.join(args.base_dir, dir_, file)
                path = path.replace('.jsonl', '')

                preprocessor.preprocess(
                    preprocessor.open_jsonl(
                        path,
                        json_key="text",
                    ),
                    save_file_name=path,
                    log_interval=100,
                )
                path = path.split('/')[-1]
                f.write(f"{path}\n")
        ff.write(f"{dir_}\n")
