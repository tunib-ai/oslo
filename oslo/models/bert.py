from transformers.models.bert.modeling_bert import BertAttention, BertOutput, BertLayer

from oslo.parallelism.mpu import LayerInfo


class BertLayerInfo(LayerInfo):
    @staticmethod
    def base():
        return BertLayer

    @staticmethod
    def attention():
        return BertAttention

    @staticmethod
    def mlp():
        return BertOutput

    @staticmethod
    def reducing_required():
        return ["all_head_size", "num_attention_heads"]
