import importlib

try:
    import transformers
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

import oslo

from oslo.torch.nn.parallel.distributed.tensor_parallel._mapping_1d import (
    Column,
    Row,
    Update,
)


class _TensorParallelMappingForHuggingFace(object):
    __MAPPING__ = {
        "Albert": [
            Column("query", "key", "value", "ffn"),
            Row("attention.dense", "ffn_output"),
            Update("num_attention_heads", "all_head_size"),
        ],
        "Bart": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
        ],
        "Bert": [
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        "Blenderbot": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
        ],
        "BlenderbotSmall": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
        ],
        "T5": [
            Column("q", "k", "v", "DenseReluDense.wi"),
            Row("o", "DenseReluDense.wo", "relative_attention_bias"),
            Update("d_model", "n_heads", "inner_dim"),
        ],
        "GPT2": [
            Column("c_attn", reverse=True, combined_qkv=True),
            Column("c_fc", "q_attn", reverse=True),
            Row("c_proj", reverse=True),
            Update("embed_dim", "split_size", "num_heads"),
        ],
        "GPTNeo": [
            Column("q_proj", "k_proj", "v_proj", "c_fc"),
            Row("out_proj", "c_proj"),
            Update("embed_dim", "num_heads"),
        ],
        "GPTJ": [
            Column("q_proj", "k_proj", "v_proj", "fc_in"),
            Row("out_proj", "fc_out"),
            Update("embed_dim", "num_attention_heads"),
        ],
        "Electra": [
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        "Roberta": [
            Column("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
    }

    def __init__(self):
        cache_mapping = {}
        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_hf_class_by_name(cls_name)
            if cls is not None:
                cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping

    @staticmethod
    def _load_hf_class_by_name(model_name):
        """
        Load base class obj by class name

        Args:
            model_name (str): model name (e.g. Bert, GPT2, T5, ...)

        Returns:
            class: XXXPreTrainedModel
        """
        try:
            transformers = importlib.import_module("transformers")
            cls = getattr(transformers, f"{model_name}PreTrainedModel", None)
            if cls is None:
                cls = getattr(transformers, f"{model_name}PretrainedModel", None)
            return cls
        except ImportError:
            return None


TENSOR_PARALLEL_MAPPING_FOR_HF = _TensorParallelMappingForHuggingFace()

HF_TO_OSLO = {
    transformers.GPT2Model: oslo.transformers.GPT2Model,
    transformers.GPT2LMHeadModel: oslo.transformers.GPT2LMHeadModel,
    transformers.GPT2DoubleHeadsModel: oslo.transformers.GPT2DoubleHeadModel,
    transformers.GPT2ForSequenceClassification: oslo.transformers.GPT2ForSequenceClassification,
    transformers.GPT2ForTokenClassification: oslo.transformers.GPT2ForTokenClassification,
}
