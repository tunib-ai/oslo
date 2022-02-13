import time

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM as ModelClass

from oslo.pytorch.kernel_fusion.compile.compilers import memory_efficient_fusion
from oslo.pytorch.kernel_fusion.graphs.register import GraphRegister

bsz, seqlen = 1, ...
model = ModelClass.from_pretrained("bert-base-uncased").cuda()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
input_tensor = tokenizer(["I am [MASK] man. How are you?"] * bsz, return_tensors="pt").to("cuda")

start = time.time()
output_1 = model(**input_tensor)
print('no fusion', time.time() - start)
print(tokenizer.batch_decode(output_1.logits.argmax(-1)))

graph = GraphRegister(model, "fuser2", memory_efficient_fusion=False)
graph.register()

start = time.time()
output_2 = model(**input_tensor)

print('fusion', time.time() - start)
print(tokenizer.batch_decode(output_2.logits.argmax(-1)))
