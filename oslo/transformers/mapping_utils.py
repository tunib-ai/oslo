import importlib

try:
    import transformers
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")

import oslo
from oslo.torch.nn.parallel.tensor_parallel import Column, Row, Update, Head
from oslo.torch.nn.parallel.expert_parallel.mapping import Front, Behind


class _ParallelMappingForHuggingFace(object):
    __MAPPING__ = {}

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

    def get_mapping(self, model):
        """
        Get mapping by model obj

        Args:
            model (PreTrainedModel): model object (e.g. BertForSequenceClassification)

        Returns:
            dict: mapping by model
        """
        mapping_by_model = None
        for cls, mapping in self.__MAPPING__.items():
            if isinstance(model, cls):
                mapping_by_model = {cls: mapping}

        assert mapping_by_model is not None, (
            f"Currently, {model.__class__.__qualname__} is not supported. "
            f"The current supported models are {list(self.__MAPPING__.keys())}"
        )
        return mapping_by_model


class _TensorParallelMappingForHuggingFace(_ParallelMappingForHuggingFace):
    __MAPPING__ = {
        "Albert": [
            Column("query", "key", "value", "ffn"),
            Column(
                "predictions.dense",
                "albert.pooler",
                "embedding_hidden_mapping_in",
                gather_output=True,
            ),
            Row("attention.dense", "ffn_output"),
            Update("num_attention_heads", "all_head_size"),
            Head(
                "predictions.decoder",
                "sop_classifier.classifier",
                "classifier",
                "qa_outputs",
            ),
        ],
        "Bart": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Column("classification_head.dense", gather_output=True),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
            Head("lm_head", "classification_head.out_proj", "qa_outputs"),
        ],
        "Bert": [
            Column("query", "key", "value", "intermediate.dense"),
            Column("pooler.dense", gather_output=True),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
            Head("decoder", "seq_relationship", "classifier", "qa_outputs"),
        ],
        "Blenderbot": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
            Head("lm_head"),
        ],
        "BlenderbotSmall": [
            Column("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
            Head("lm_head"),
        ],
        "T5": [
            Column("q", "k", "v", "DenseReluDense.wi"),
            Row("o", "DenseReluDense.wo", "relative_attention_bias"),
            Update("d_model", "n_heads", "inner_dim"),
            Head("lm_head"),
        ],
        "GPT2": [
            Column("c_attn", reversed=True, combined_qkv=True),
            Column("c_fc", "q_attn", reversed=True),
            Row("c_proj", reversed=True),
            Update("embed_dim", "split_size", "num_heads"),
            Head("lm_head", "score", "classifier", "summary"),
        ],
        "GPTNeo": [
            Column("q_proj", "k_proj", "v_proj", "c_fc"),
            Row("out_proj", "c_proj"),
            Update("embed_dim", "num_heads"),
            Head("lm_head", "score", "qa_outputs"),
        ],
        "GPTJ": [
            Column("q_proj", "k_proj", "v_proj", "fc_in"),
            Row("out_proj", "fc_out"),
            Update("embed_dim", "num_attention_heads"),
            Head("lm_head", "score"),
        ],
        "Electra": [
            Column("query", "key", "value", "intermediate.dense"),
            Column(
                "electra.embeddings_project",
                "classifier.dense",
                "discriminator_predictions.dense",
                "generator_predictions.dense",
                gather_output=True,
            ),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
            Head(
                "generator_lm_head",
                "classifier.out_proj",
                "discriminator_predictions.dense_prediction",
                "classifier",
                "qa_outputs",
                "summary",
            ),
        ],
        "Roberta": [
            Column("query", "key", "value", "intermediate.dense"),
            Column(
                "lm_head.dense",
                "classifier.dense",
                "roberta.pooler",
                gather_output=True,
            ),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
            Head("lm_head.decoder", "classifier.out_proj", "classifier", "qa_outputs"),
        ],
    }

    def __init__(self):
        cache_mapping = {}
        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_hf_class_by_name(cls_name)
            if cls is not None:
                cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping


class _ExpertParallelMappingForHuggingFace(_ParallelMappingForHuggingFace):
    __MAPPING__ = {
        "Albert": [
            Front("ffn", layer="albert_layer_groups.albert_layers", enc_name="encoder"),
            Behind(
                "ffn_output",
                layer="albert_layer_groups.albert_layers",
                enc_name="encoder",
            ),
        ],
        "Bart": [
            Front(
                "fc1",
                layer="layers",
                enc_name="encoder",
                dec_name="decoder",
            ),
            Behind("fc2", layer="layers", enc_name="encoder", dec_name="decoder"),
        ],
        "Bert": [
            Front("intermediate.dense", layer="layer", enc_name="encoder"),
            Behind("output.dense", layer="layer", enc_name="encoder"),
        ],
        "Blenderbot": [
            Front("fc1", layer="layers", enc_name="encoder", dec_name="decoder"),
            Behind("fc2", layer="layers", enc_name="encoder", dec_name="decoder"),
        ],
        "BlenderbotSmall": [
            Front(
                "fc1",
                layer="layers",
                enc_name="encoder",
                dec_name="decoder",
            ),
            Behind("fc2", layer="layers", enc_name="encoder", dec_name="decoder"),
        ],
        "T5": [
            Front(
                "DenseReluDense.wi",
                layer="block",
                enc_name="encoder",
                dec_name="decoder",
            ),
            Behind(
                "DenseReluDense.wo",
                layer="block",
                enc_name="encoder",
                dec_name="decoder",
            ),
        ],
        "GPT2": [
            Front(
                "mlp.c_fc",
                reverse=True,
                layer="h",
                dec_name="h",
            ),
            Behind(
                "mlp.c_proj",
                reverse=True,
                layer="h",
                dec_name="h",
            ),
        ],
        "GPTNeo": [
            Front("c_fc", layer="h", dec_name="h"),
            Behind("c_proj", layer="h", dec_name="h"),
        ],
        "GPTJ": [
            Front("fc_in", layer="h", dec_name="h"),
            Behind("fc_out", layer="h", dec_name="h"),
        ],
        "Electra": [
            Front("intermediate.dense", layer="layer", enc_name="encoder"),
            Behind("output.dense", layer="layer", enc_name="encoder"),
        ],
        "Roberta": [
            Front("intermediate.dense", layer="layer", enc_name="encoder"),
            Behind("output.dense", layer="layer", enc_name="encoder"),
        ],
    }

    def __init__(self):
        cache_mapping = {}
        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_hf_class_by_name(cls_name)
            if cls is not None:
                cache_mapping[cls] = mapping

        self.__MAPPING__ = cache_mapping


HF_TO_OSLO = {
    transformers.GPT2Model: oslo.transformers.GPT2Model,
    transformers.GPT2LMHeadModel: oslo.transformers.GPT2LMHeadModel,
    transformers.GPT2DoubleHeadsModel: oslo.transformers.GPT2DoubleHeadModel,
    transformers.GPT2ForSequenceClassification: oslo.transformers.GPT2ForSequenceClassification,
    transformers.GPT2ForTokenClassification: oslo.transformers.GPT2ForTokenClassification,
}
