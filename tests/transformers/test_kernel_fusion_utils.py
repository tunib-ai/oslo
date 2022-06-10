from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
)

from oslo.transformers.kernel_fusion_utils import fused_no_repeat_ngram_logits_processor

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to("cuda")

output = model.generate(
    **tokenizer("hello", return_tensors="pt").to("cuda"), no_repeat_ngram_size=2
)
print(tokenizer.decode(output[0]))

fused_no_repeat_ngram_logits_processor(model)

output = model.generate(
    **tokenizer("hello", return_tensors="pt").to("cuda"), no_repeat_ngram_size=2
)
print(tokenizer.decode(output[0]))
