from oslo.transformers.tasks.data_causal_lm import (
    ProcessorForCausalLM,
    DataCollatorForCausalLM,
)
from oslo.transformers.tasks.data_masked_lm import (
    ProcessorForMaskedLM,
    DataCollatorForMaskedLM,
)
from oslo.transformers.tasks.data_sequence_classification import (
    ProcessorForSequenceClassification,
    DataCollatorForSequenceClassification,
)
from oslo.transformers.tasks.data_token_classification import (
    ProcessorForTokenClassification,
    DataCollatorForTokenClassification,
)
from oslo.transformers.tasks.data_summarization import (
    ProcessorForSummarization,
    DataCollatorForSummarization,
)
from oslo.transformers.tasks.data_bert_pretraining import (
    ProcessorForBertPretraining,
    DataCollatorForBertPretraining,
)
from oslo.transformers.tasks.data_albert_pretraining import (
    ProcessorForAlbertPretraining,
    DataCollatorForAlbertPretraining,
)
from oslo.transformers.tasks.data_bart_pretraining import (
    ProcessorForBartPretraining,
    DataCollatorForBartPretraining,
)
from oslo.transformers.tasks.data_t5_pretraining import (
    ProcessorForT5Pretraining,
    DataCollatorForT5Pretraining,
)
