from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP

from oslo.parallelism.mpu import LayerInfo


class GPT2LayerInfo(LayerInfo):
    @staticmethod
    def base():
        return GPT2Block

    @staticmethod
    def attention():
        return GPT2Attention

    @staticmethod
    def mlp():
        return GPT2MLP

    @staticmethod
    def reducing_required():
        return ["embed_dim", "split_size", "num_heads"]
