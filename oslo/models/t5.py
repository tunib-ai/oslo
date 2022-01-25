from transformers.models.t5.modeling_t5 import T5Block, T5Attention, T5DenseReluDense

from oslo.parallelism.mpu import LayerInfo


class T5LayerInfo(LayerInfo):
    @staticmethod
    def base():
        return T5Block

    @staticmethod
    def attention():
        return T5Attention

    @staticmethod
    def mlp():
        return T5DenseReluDense

    @staticmethod
    def reducing_required():
        return ["d_model", "n_heads", "inner_dim"]
