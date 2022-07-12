"""PyTorch BLOOM model."""

import math
from typing import Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

import oslo.torch.nn as onn
import oslo.torch.nn.modules.functional as F
from oslo.transformers.modeling_utils import OsloModel

try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
        CausalLMOutputWithCrossAttentions,
        SequenceClassifierOutputWithPast,
        TokenClassifierOutput,
    )
    from transformers.modeling_utils import PreTrainedModel
    from transformers.utils import logging
    from transformers.models.bloom.configuration_bloom import BloomConfig
except ImportError:
    print("You have to install `transformers` to use `oslo.transformers` modules")


logger = logging.get_logger(__name__)


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.

    Args:
        tensor: ([`torch.tensor`], *required*):
            input tensor to split
        num_partitions ([`int`], *required*):
            number of partitions to split the tensor
        contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
            If True, make each chunk contiguous in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    numerator, denominator = tensor.size()[last_dim], num_partitions
    if not (numerator % denominator == 0):
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    last_dim_size = numerator // denominator
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def attention_mask_func(attention_scores, attention_mask, causal_mask):
    if attention_mask.dtype == torch.bool:
        attention_mask_bool = ~attention_mask
    else:
        attention_mask_bool = (1 - attention_mask).bool()

    query_length, key_length, n_heads = (
        attention_scores.size(2),
        attention_scores.size(3),
        attention_scores.size(1),
    )
    padded_causal_mask = (
        attention_mask_bool[:, None, key_length - query_length : key_length, None]
        + ~causal_mask[:, :, key_length - query_length : key_length, :key_length]
    ).bool()
    padded_causal_mask = (
        padded_causal_mask + attention_mask_bool[:, None, None, :key_length].bool()
    )
    # Make use of floats
    return (
        attention_scores.masked_fill_(
            padded_causal_mask.expand(-1, n_heads, -1, -1), -10000.0
        ),
        padded_causal_mask,
    )


def build_alibi_tensor(max_seq_len, n_head, dtype=torch.bfloat16):
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    Args:
    Returns tensor shaped (n_head, 1, max_seq_len)
        max_seq_len: (`int`, *required*):
            max sequence length
        n_head: (`int`, *required*):
            number of heads
        dtype: (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(n_head)).unsqueeze(1).unsqueeze(1)
    arange_tensor = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
    alibi = slopes * arange_tensor.expand(n_head, -1, -1)

    alibi = alibi.to(dtype)

    return alibi


def pre_process_alibi_for_pad(alibi, attention_mask, num_heads):
    """
    Args:
    Pre-process the alibi tensor for padding.
        alibi: ([`torch.tensor`], *required*):
            alibi tensor to pre-process
        attention_mask: ([`torch.tensor`], *required*):
            attention mask to pre-process"""

    # Sanity check if we are not inferring less tokens than the total sequence length
    # This usually happens when the inference is done with past_key_values
    # In this case we re-create the alibi tensor with the correct sequence length
    if attention_mask.shape[-1] != alibi.shape[-1]:
        alibi = build_alibi_tensor(
            attention_mask.shape[-1], num_heads, alibi.dtype
        ).repeat(attention_mask.shape[0], 1, 1)
    # Get the indexes of the padding tokens
    index_x0, index_y0 = torch.where(attention_mask == 0.0)
    index_x1, index_y1 = torch.where(attention_mask == 1.0)

    # Clone the embeddings  - we can detach because the embeddings are not learned
    # Get a reference tensor
    slice_reference_alibi = build_alibi_tensor(alibi.shape[-1], num_heads, alibi.dtype)

    # Loop over the batch where the padding is and replace the alibi tensor by the reference tensor
    # Only where you do not have padding. Replace padding tokens by zeros
    # This operation can be seen as a shifting operation.
    for i, index in enumerate(torch.unique(index_x0)):
        slice_to_modify = torch.zeros_like(slice_reference_alibi)
        index_shift = index_y1[index_x1 == index]
        shift_value = len(index_shift)
        slice_to_modify[:, :, index_shift] = slice_reference_alibi[:, :, :shift_value]
        alibi[index * num_heads : (index + 1) * num_heads] = slice_to_modify
    return alibi


def dropout_add(x, residual, prob, training):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def fused_bias_dropout_add(x, bias, residual, prob, training, inplace=False):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        bias (`torch.tensor`, *required*)
            bias tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.fused_bias_dropout(x, bias, p=prob, training=training, inplace=inplace)
    out = residual + out
    return out


class BloomScaledSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16 (`bool`, *required*):
            flag to indicate if input in fp16 data format.
        input_in_bf16 (`bool`, *required*):
            flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion (`bool`, *required*):
            flag to indicate user want to use softmax fusion
        mask_func (`function`, *required*):
            mask function to be applied.
        softmax_in_fp32 (`bool`, *required*):
            if true, softmax in performed at fp32 precision.
        scale (`float`, *required*):
            scaling factor used in input tensor scaling.
    """

    def __init__(self, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__()
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        if not (self.scale is None or softmax_in_fp32):
            raise ValueError("softmax should be in fp32 when scaled")

    def forward(self, input, mask, max_positions):
        input_dtype = input.dtype
        input_in_16bit = input_dtype in [torch.float16, torch.bfloat16]
        softmax_dtype = torch.float32 if self.softmax_in_fp32 else input_dtype

        if self.scale is not None:
            input = input * self.scale

        if mask is not None:
            mask = mask.to(input.device)
            causal_mask = (
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
                .view(1, 1, max_positions, max_positions)
                .to(input.device)
            )
            mask_output, padded_causal_mask = self.mask_func(input, mask, causal_mask)
            probs = nn.functional.softmax(mask_output, dim=-1, dtype=softmax_dtype) * (
                ~padded_causal_mask
            )
        else:
            probs = nn.functional.softmax(input, dim=-1, dtype=softmax_dtype)

        if input_in_16bit and self.softmax_in_fp32:
            probs = probs.to(dtype=input_dtype)

        return probs


class BloomAttention(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.masked_softmax_fusion = config.masked_softmax_fusion
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim) * self.layer_number

        # Scaled Softmax
        self.scale_mask_softmax = BloomScaledSoftmax(
            self.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            self.layer_number,
        )

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = onn.Linear(self.hidden_size, self.hidden_size, skip_bias_add=True)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        residual,
        layer_past=None,
        attention_mask=None,
        alibi=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # repeat alibi tensor with the batch size
        alibi = alibi.repeat(hidden_states.shape[0], 1, 1).to(hidden_states.device)

        # apply preprocessing if the input is padded
        if attention_mask is not None and 0 in attention_mask:
            alibi = pre_process_alibi_for_pad(alibi, attention_mask, self.num_heads)

        mixed_x_layer = self.query_key_value(hidden_states)

        # [batch_size, seq_length, 3 x hidden_size] --> [batch_size, seq_length, num_heads, 3 x head_dim]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_heads,
            3 * self.head_dim,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [batch_size, seq_length, num_heads, 3 x head_dim] --> 3  [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
            mixed_x_layer, 3
        )

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=1)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=1
            )

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size, head_dim, q_length, k_length]
        output_size = (
            query_layer.size(0),
            query_layer.size(2),
            query_layer.size(1),
            key_layer.size(1),
        )

        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.transpose(1, 0).reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )

        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.transpose(1, 0).reshape(
            output_size[3], output_size[0] * output_size[1], -1
        )

        # slice alibi tensor until the query length
        sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]

        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        beta = 1.0 / self.layer_number

        matmul_result = torch.baddbmm(
            sliced_alibi,
            query_layer.transpose(1, 0),
            key_layer.transpose(1, 0).transpose(1, 2),
            beta=beta,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.view(*output_size)
        if F._is_fused_scale_mask_softmax_available(
            input=attention_scores,
            scale=1.0,
            use_triang_mask=True,
            softmax_in_fp32=self.attention_softmax_in_fp32,
        ):
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(
                    dtype=attention_scores.dtype
                )  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -10000.0

            attention_probs = F._fused_scale_mask_softmax_cuda(
                input=attention_scores,
                scale=1.0,
                use_triang_mask=True,
                pad_mask=attention_mask,
            )
        else:
            # attention scores and attention mask [b, np, sq, sk]
            max_positions = max(attention_scores.shape[-1], attention_scores.shape[-2])
            attention_probs = self.scale_mask_softmax(
                attention_scores, attention_mask, max_positions
            ).to(value_layer.dtype)

        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context layer shape: [batch_size, num_heads, q_length, head_dim]
        output_size = (
            value_layer.size(0),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [k_length, batch_size x num_heads, head_dim]
        value_layer = value_layer.transpose(1, 0).reshape(
            value_layer.size(1), output_size[0] * output_size[1], -1
        )

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.view(*output_size)

        # [batchs_size, num_heads, q_length, head_dim] --> [q_length, batch_size, num_heads, head_dim]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [q_length, batch_size, num_heads, head_dim] --> [q_length, batch_size, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [q_length, batch_size, hidden_size]

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = context_layer.shape[-1] / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + nn.functional.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
            output = output_tensor.transpose(1, 0)
            output = dropout_add(output, residual, self.hidden_dropout, self.training)
        else:
            output_tensor, bias = self.dense(context_layer)
            output = output_tensor.transpose(
                1, 0
            )  # [batch_size, q_length, hidden_size]
            output = fused_bias_dropout_add(
                output, bias, residual, self.hidden_dropout, self.training
            )

        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = onn.Linear(
            hidden_size, 4 * hidden_size, skip_bias_add=True
        )
        self.dense_4h_to_h = onn.Linear(
            4 * hidden_size, hidden_size, skip_bias_add=True
        )
        self.hidden_dropout = config.hidden_dropout
        self.gelu_impl = onn.FusedBiasGeLU()

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(*self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + nn.functional.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[
                        :, int(i * slices) : int((i + 1) * slices)
                    ],
                )
            output = dropout_add(
                intermediate_output, residual, self.hidden_dropout, self.training
            )
        else:
            intermediate_output, bias = self.dense_4h_to_h(hidden_states)
            output = fused_bias_dropout_add(
                intermediate_output, bias, residual, self.hidden_dropout, self.training
            )

        return output


class BloomBlock(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.n_head = config.n_head
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PreTrainedModel, OsloModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BloomBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, onn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BloomModel):
            module.gradient_checkpointing = value


class BloomModel(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.n_head = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.word_embeddings_layernorm = LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [
                BloomBlock(config, layer_number=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_head x N x N
        # head_mask has shape n_layer x batch x n_head x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        current_sequence_length = hidden_states.shape[1]
        if past_key_values[0] is not None:
            current_sequence_length += past_key_values[0][0].shape[1]
        alibi = build_alibi_tensor(
            current_sequence_length, self.n_head, hidden_states.dtype
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions, alibi)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = hidden_states.view(output_shape)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class BloomForSequenceClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BloomForTokenClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = BloomModel(config)
        if (
            hasattr(config, "classifier_dropout")
            and config.classifier_dropout is not None
        ):
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
