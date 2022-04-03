from abc import ABCMeta
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers.models.gpt2.modeling_gpt2 import *


class RingQK(metaclass=ABCMeta):
    @staticmethod
    @custom_fwd
    def forward():
        pass

    @staticmethod
    @custom_bwd
    def backward():
        pass


class RingAV(metaclass=ABCMeta):
    @staticmethod
    @custom_fwd
    def forward():
        pass

    @staticmethod
    @custom_bwd
    def backward():
        pass


class GPT2WithSPLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        self.transformer = GPT2WithSPModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


class GPT2WithSPModel(GPT2Model):
    def __init__(self, config):
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2WithSPBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GPT2WithSPBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


class GPT2RingAttention(GPT2Attention):
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        if self.reorder_and_upcast_attn:
            raise NotImplementedError(
                "Not implemented for Sequence Parallelism with `reorder_and_upcast_attn=True`"
            )
        batch_size = query.size(0)
        head_count = query.size(1)
        sub_seq_length = query.size(2)
        query = query.view(batch_size * head_count, sub_seq_length, -1)
        key = key.view(batch_size * head_count, sub_seq_length, -1)
        attn_weights = RingQK.apply(
            query.contiguous(),  # [batch_size * num_heads, sub_seq_len, head_size]
            key.contiguous(),  # [batch_size * num_heads, sub_seq_len, head_size],
            batch_size,
            self.num_heads,
            sub_seq_length
        )

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value = value.view(batch_size * head_count, sub_seq_length, -1)
        attn_output = RingAV.apply(
            attn_weights,
            value.contiguous(),
            batch_size,
            self.num_heads,
            self.head_dim,
            sub_seq_length
        )

        return attn_output, attn_weights
