import torch
import torch.nn as nn
from sp import TransformerSelfAttentionRingTest
from colossalai.kernel.jit import bias_dropout_add_fused_train, bias_dropout_add_fused_inference
from colossalai.kernel.cuda_native import LayerNorm
from .mlp import TransformerMLP
from .dropout import get_bias_dropout_add


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class BertLayer(nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self,
                 layer_number,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout,
                 mlp_ratio,
                 hidden_dropout,
                 is_naive_fp16,
                 apply_residual_connection_post_layernorm=False,
                 fp32_residual_connection=False,
                 bias_dropout_fusion: bool = True,
                 convert_fp16_to_fp32_in_softmax: bool = False):
        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.fp32_residual_connection = fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size)
        self.input_layernorm = self.input_layernorm.float()

        # Self attention.
        self.self_attention = TransformerSelfAttentionRingTest(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            attention_mask_func=attention_mask_func,
            layer_number=layer_number,
            apply_query_key_layer_scaling=True,
            convert_fp16_to_fp32_in_softmax=convert_fp16_to_fp32_in_softmax,
            fp16=is_naive_fp16
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(hidden_size)
        self.post_attention_layernorm = self.post_attention_layernorm.float()

        self.mlp = TransformerMLP(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states, attention_mask, ks_save_path=None):
        # hidden_states: [batch_size, sub_seq_len, hidden_size]
        # attention_mask: [batch_size, 1, sub_seq_len, seq_len]

        # to check
        mid_outputs = []

        # Layer norm at the beginning of the transformer layer.

        hidden_states = hidden_states.float()

        layernorm_output = self.input_layernorm(hidden_states)

        layernorm_output = layernorm_output.type(torch.float64)

        mid_outputs.append(layernorm_output.detach())

        # Self attention.
        attention_output, attention_bias, attention_mid_outputs = self.self_attention(layernorm_output, attention_mask, ks_save_path)

        torch.distributed.barrier()

        mid_outputs += [o.detach() for o in attention_mid_outputs]
        mid_outputs.append(attention_output.detach())
        mid_outputs.append(attention_bias.detach())

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        mid_outputs.append(residual.detach())

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        mid_outputs.append(layernorm_input.detach())

        layernorm_input = layernorm_input.float()

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        layernorm_output = layernorm_output.type(torch.float64)

        mid_outputs.append(layernorm_output.detach())

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        mid_outputs.append(mlp_output.detach())
        mid_outputs.append(mlp_bias.detach())

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        mid_outputs.append(residual.detach())

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        return output, mid_outputs
